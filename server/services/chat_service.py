import time
import re
import traceback
from typing import Dict, List, AsyncGenerator
from fastapi import HTTPException
import json
import asyncio

from services.agent_service import agent_manager
from services.kg_service import extract_kg_from_message
from utils.concurrent import chat_manager, feedback_manager


async def process_chat(message: str, session_id: str, debug: bool = False, agent_type: str = "hybrid_agent", 
                       use_deeper_tool: bool = True, show_thinking: bool = False) -> Dict:
    """
    处理聊天请求
    
    Args:
        message: 用户消息
        session_id: 会话ID
        debug: 是否为调试模式
        agent_type: Agent类型
        use_deeper_tool: 是否使用增强版研究工具 (for deep_research_agent)
        show_thinking: 是否显示思考过程 (for deep_research_agent)
        
    Returns:
        Dict: 聊天响应结果
    """
    # 生成锁的键
    lock_key = f"{session_id}_chat"
    
    # 非阻塞方式尝试获取锁
    lock_acquired = chat_manager.try_acquire_lock(lock_key)
    if not lock_acquired:
        # 如果无法获取锁，说明有另一个请求正在处理
        raise HTTPException(
            status_code=429, 
            detail="当前有其他请求正在处理，请稍后再试"
        )
    
    try:
        # 更新操作时间戳
        chat_manager.update_timestamp(lock_key)
        
        # 获取指定的agent
        try:
            selected_agent = agent_manager.get_agent(agent_type)
            if agent_type == "deep_research_agent":
                selected_agent.is_deeper_tool(use_deeper_tool)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 首先尝试快速路径 - 跳过完整处理
        try:
            start_fast = time.time()
            fast_result = selected_agent.check_fast_cache(message, session_id)
            
            if fast_result:
                print(f"API快速路径命中: {time.time() - start_fast:.4f}s")
                
                # 在调试模式下，需要提供额外信息
                if debug:
                    # 提供模拟的执行日志
                    mock_log = [{
                        "node": "fast_cache_hit", 
                        "timestamp": time.time(), 
                        "input": message, 
                        "output": "高质量缓存命中，跳过完整处理"
                    }]
                    
                    # 尝试提取图谱数据，对deep_research_agent禁用
                    kg_data = {"nodes": [], "links": []}
                    if agent_type != "deep_research_agent":
                        try:
                            kg_data = extract_kg_from_message(fast_result)
                        except:
                            kg_data = {"nodes": [], "links": []}
                        
                    return {
                        "answer": fast_result,
                        "execution_log": mock_log,
                        "kg_data": kg_data
                    }
                else:
                    # 标准模式直接返回答案
                    return {"answer": fast_result}
        except Exception as e:
            # 快速路径失败，继续常规流程
            print(f"快速路径检查失败: {e}")
        
        # 检查是否为deep_research_agent且是否显示思考过程
        show_thinking = agent_type == "deep_research_agent"
        
        if debug:
            # 在Debug模式下使用ask_with_trace或ask_with_thinking，并返回知识图谱数据
            if agent_type == "deep_research_agent":
                # 使用ask_with_thinking方法获取带思考过程的结果
                result = selected_agent.ask_with_thinking(message, thread_id=session_id)
                
                # 从结果字典中获取各个组件
                thinking_process = result.get("thinking_process", "")
                answer_content = result.get("answer", "")
                retrieved_info = result.get("retrieved_info", [])
                reference = result.get("reference", {})
                execution_logs = result.get("execution_logs", [])  # 获取执行日志
                
                # 为deep_research_agent禁用知识图谱数据
                kg_data = {"nodes": [], "links": []}
                
                # 提取迭代轮次信息以便前端展示
                iterations = extract_iterations(retrieved_info)
                
                # 如果未能从retrieved_info提取到有效迭代，尝试从thinking_process中提取
                if not iterations or len(iterations) == 0:
                    print("从retrieved_info中没有提取到迭代信息，尝试从thinking_process中提取")
                    thinking_iterations = extract_iterations_from_thinking(thinking_process)
                    if thinking_iterations and len(thinking_iterations) > 0:
                        print(f"从thinking_process中提取到{len(thinking_iterations)}轮迭代")
                        iterations = thinking_iterations
                
                # 构建执行日志
                execution_log = [{
                    "node": "deep_research", 
                    "input": message, 
                    "output": "\n".join(execution_logs) if execution_logs else "无执行日志"
                }]
                
                # 打印日志长度信息
                logs_count = len(execution_logs)
                print(f"执行日志数量: {logs_count}条")
                
                # 构建完整响应，包含执行日志
                return {
                    "answer": answer_content,
                    "execution_log": execution_log,
                    "kg_data": kg_data,
                    "reference": reference,
                    "iterations": iterations,
                    "raw_thinking": thinking_process,
                    "execution_logs": execution_logs,
                }
            else:
                # 其他Agent使用标准的ask_with_trace
                result = selected_agent.ask_with_trace(
                    message, 
                    thread_id=session_id,
                )
                
                # 从结果中提取知识图谱数据
                kg_data = extract_kg_from_message(result["answer"])
                
                return {
                    "answer": result["answer"],
                    "execution_log": result["execution_log"],
                    "kg_data": kg_data,
                }
        else:
            # 标准模式
            if agent_type == "deep_research_agent" and show_thinking:
                # 使用ask_with_thinking方法获取带思考过程的结果
                result = selected_agent.ask_with_thinking(message, thread_id=session_id)
                
                # 从结果字典中获取各个组件
                thinking_process = result.get("thinking_process", "")
                answer_content = result.get("answer", "")
                execution_logs = result.get("execution_logs", [])
                
                # 返回思考过程、答案和执行日志
                return {
                    "answer": answer_content,
                    "raw_thinking": thinking_process,
                    "execution_logs": execution_logs
                }
            else:
                # 普通模式，使用标准ask方法
                # 检查是否为DeepResearchAgent类型，只有DeepResearchAgent支持show_thinking参数
                if agent_type == "deep_research_agent":
                    answer = selected_agent.ask(
                        message, 
                        thread_id=session_id,
                        show_thinking=show_thinking # deep_research_agent支持此参数
                    )
                else:
                    # 其他Agent类型不支持show_thinking参数
                    answer = selected_agent.ask(
                        message, 
                        thread_id=session_id
                    )
                return {"answer": answer}
    except Exception as e:
        error_msg = str(e)
        print(f"处理聊天请求时出错: {error_msg}")
        traceback.print_exc()

        # 检查是否是向量索引不存在的错误
        if "vector index" in error_msg.lower() and "does not exist" in error_msg.lower():
            friendly_msg = (
                "知识图谱尚未构建，请先完成以下步骤：\n"
                "1. 进入「📚 文档管理」上传文档\n"
                "2. 进入「🏗️ 构建管理」点击「全量构建」\n"
                "3. 等待构建完成后再进行查询\n\n"
                "注意：首次构建可能需要几分钟时间，具体取决于文档数量。"
            )
            raise HTTPException(status_code=400, detail=friendly_msg)

        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # 释放锁
        chat_manager.release_lock(lock_key)
        
        # 清理过期的锁
        chat_manager.cleanup_expired_locks()

async def process_chat_stream(
    message: str, 
    session_id: str, 
    debug: bool = False, 
    agent_type: str = "hybrid_agent",
    use_deeper_tool: bool = True, 
    show_thinking: bool = False
) -> AsyncGenerator[str, None]:
    """
    处理聊天请求，返回流式输出
    
    Args:
        message: 用户消息
        session_id: 会话ID
        debug: 是否为调试模式
        agent_type: Agent类型
        use_deeper_tool: 是否使用增强版研究工具
        show_thinking: 是否显示思考过程
        
    Yields:
        流式文本块或状态更新
    """
    # 生成锁的键
    lock_key = f"{session_id}_chat"
    
    # 非阻塞方式尝试获取锁
    lock_acquired = chat_manager.try_acquire_lock(lock_key)
    if not lock_acquired:
        # 返回错误流
        yield json.dumps({"status": "error", "message": "当前有其他请求正在处理，请稍后再试"})
        return
    
    try:
        # 更新操作时间戳
        chat_manager.update_timestamp(lock_key)
        
        # 获取指定的agent
        try:
            selected_agent = agent_manager.get_agent(agent_type)
            if agent_type == "deep_research_agent":
                selected_agent.is_deeper_tool(use_deeper_tool)
        except ValueError as e:
            yield json.dumps({"status": "error", "message": str(e)})
            return
        
        # 首先尝试快速路径缓存
        try:
            start_fast = time.time()
            fast_result = selected_agent.check_fast_cache(message, session_id)
            
            if fast_result:
                print(f"API快速路径命中: {time.time() - start_fast:.4f}s")
                # 如果是调试模式，生成模拟执行日志
                if debug:
                    mock_log = {
                        "node": "fast_cache_hit", 
                        "timestamp": time.time(), 
                        "input": message, 
                        "output": "高质量缓存命中，跳过完整处理"
                    }
                    yield {"execution_log": mock_log}
                
                yield json.dumps({"status": "token", "content": fast_result})
                yield json.dumps({"status": "done"})
                return
        except Exception as e:
            print(f"快速路径检查失败: {e}")
        
        # 保存执行轨迹（针对调试模式）
        execution_log = []
        
        # 对于深度研究Agent使用思考流
        if agent_type == "deep_research_agent" and show_thinking:
            # 获取思考过程的流处理
            thinking_step = False
            thinking_content = ""
            
            async for chunk in selected_agent.ask_stream(message, thread_id=session_id):
                if isinstance(chunk, dict):
                    # 字典形式包含状态信息
                    if "execution_log" in chunk and debug:
                        execution_log.append(chunk["execution_log"])
                        yield {"execution_log": chunk["execution_log"]}
                    else:
                        yield chunk
                elif "[深度研究]" in chunk or "[KB检索]" in chunk:
                    # 这是思考步骤
                    thinking_step = True
                    thinking_content += chunk
                    yield json.dumps({"status": "thinking", "content": chunk})
                else:
                    # 正常内容
                    if thinking_step:
                        thinking_step = False
                        yield json.dumps({"status": "answer_start"})
                    
                    yield json.dumps({"status": "token", "content": chunk})
            
            # 发送完成消息
            yield json.dumps({"status": "done", "thinking_content": thinking_content})
            
            return
        
        # 对于其他Agent类型，使用标准流式处理
        if agent_type in ["hybrid_agent", "graph_agent", "naive_rag_agent"]:
            # 为调试模式收集执行轨迹
            if debug:
                # 首先获取执行轨迹
                trace_result = await asyncio.to_thread(
                    selected_agent.ask_with_trace,
                    message,
                    thread_id=session_id
                )
                
                # 发送执行轨迹
                if "execution_log" in trace_result:
                    for log_entry in trace_result["execution_log"]:
                        yield {"execution_log": log_entry}
                        execution_log.append(log_entry)
                
                # 发送答案，模拟流式输出
                answer = trace_result["answer"]
                chunk_size = 10  # 每个块的字符数
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i:i+chunk_size]
                    yield json.dumps({"status": "token", "content": chunk})
                    await asyncio.sleep(0.01)  # 小延迟模拟流式输出
            else:
                # 使用Agent的流式接口
                async for chunk in selected_agent.ask_stream(message, thread_id=session_id):
                    yield json.dumps({"status": "token", "content": chunk})
            
            # 发送完成消息
            yield json.dumps({"status": "done"})
        else:
            # 对于不支持流式处理的Agent，回退到非流式处理并模拟流
            if debug:
                # 首先获取执行轨迹
                trace_result = await asyncio.to_thread(
                    selected_agent.ask_with_trace,
                    message,
                    thread_id=session_id
                )
                
                # 发送执行轨迹
                if "execution_log" in trace_result:
                    for log_entry in trace_result["execution_log"]:
                        yield {"execution_log": log_entry}
                        execution_log.append(log_entry)
                
                # 发送答案，模拟流式输出
                answer = trace_result["answer"]
                chunk_size = 10  # 每个块的字符数
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i:i+chunk_size]
                    yield json.dumps({"status": "token", "content": chunk})
                    await asyncio.sleep(0.01)  # 小延迟模拟流式输出
            else:
                # 非调试模式，简单获取答案
                answer = selected_agent.ask(message, thread_id=session_id)
                
                # 分块发送响应以模拟流式输出
                chunk_size = 10  # 每个块的字符数
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i:i+chunk_size]
                    yield json.dumps({"status": "token", "content": chunk})
                    await asyncio.sleep(0.01)  # 小延迟模拟流式输出
            
            # 发送完成消息
            yield json.dumps({"status": "done"})
            
    except Exception as e:
        error_msg = str(e)
        print(f"处理聊天请求时出错: {error_msg}")
        print(traceback.format_exc())

        # 检查是否是向量索引不存在的错误
        if "vector index" in error_msg.lower() and "does not exist" in error_msg.lower():
            friendly_msg = (
                "知识图谱尚未构建，请先完成以下步骤：\n"
                "1. 进入「📚 文档管理」上传文档\n"
                "2. 进入「🏗️ 构建管理」点击「全量构建」\n"
                "3. 等待构建完成后再进行查询\n\n"
                "注意：首次构建可能需要几分钟时间，具体取决于文档数量。"
            )
            yield json.dumps({"status": "error", "message": friendly_msg})
        else:
            yield json.dumps({"status": "error", "message": error_msg})
    finally:
        # 释放锁
        chat_manager.release_lock(lock_key)
        
        # 清理过期的锁
        chat_manager.cleanup_expired_locks()


def extract_iterations(retrieved_info):
    """
    从检索信息中提取迭代轮次
    
    Args:
        retrieved_info: 检索到的信息列表
        
    Returns:
        List: 迭代轮次信息
    """
    if not retrieved_info:
        return []
    
    # 根据DeepResearchTool的输出，retrieved_info可能是一个列表
    if isinstance(retrieved_info, list):
        # 合并所有检索信息为一个字符串
        text_items = []
        for item in retrieved_info:
            if isinstance(item, str):
                text_items.append(item)
            else:
                try:
                    text_items.append(str(item))
                except:
                    pass
        
        full_text = "\n".join(text_items)
    else:
        # 如果是字符串或其他类型，转换为字符串
        full_text = str(retrieved_info)
    
    # 将文本按照迭代轮次分割
    iterations = []
    current_iteration = {"round": 1, "content": [], "queries": []}
    
    lines = full_text.split('\n')
    for line in lines:
        # 检测迭代轮次开始
        round_match = re.search(r'\[深度研究\]\s*开始第(\d+)轮迭代', line)
        if round_match:
            # 如果已有内容，保存前一轮
            if current_iteration["content"]:
                iterations.append(current_iteration)
            
            # 开始新一轮
            round_num = int(round_match.group(1))
            current_iteration = {"round": round_num, "content": [line], "queries": []}
        # 检测查询
        elif re.search(r'\[深度研究\]\s*执行查询:', line):
            query = re.sub(r'\[深度研究\]\s*执行查询:\s*', '', line).strip()
            current_iteration["queries"].append(query)
            current_iteration["content"].append(line)
        # 检测是否发现有用信息
        elif re.search(r'\[深度研究\]\s*发现有用信息:', line):
            current_iteration["content"].append(line)
            info = re.sub(r'\[深度研究\]\s*发现有用信息:\s*', '', line).strip()
            current_iteration["useful_info"] = info
        # 其他行
        else:
            current_iteration["content"].append(line)
    
    # 添加最后一轮
    if current_iteration["content"]:
        iterations.append(current_iteration)
    
    # 如果没有生成有效的迭代，创建一个基本迭代
    if not iterations:
        # 从原始文本中提取有用信息
        useful_info = None
        queries = []
        
        # 查找最终信息
        for i, line in enumerate(lines):
            if "Final Information" in line and i + 1 < len(lines):
                useful_info = lines[i + 1]
                break
            # 查找可能的查询
            if ">" in line and "?" in line:
                query = line.strip("> ").strip()
                if query and query not in queries:
                    queries.append(query)
        
        return [{
            "round": 1,
            "content": lines,
            "queries": queries if queries else ["原始查询"],
            "useful_info": useful_info or "深度研究已完成，但无法提取详细迭代信息"
        }]
    
    return iterations

def extract_iterations_from_thinking(thinking_process: str) -> List[Dict]:
    """
    从思考过程中提取迭代信息
    
    Args:
        thinking_process: 思考过程文本
        
    Returns:
        List[Dict]: 迭代轮次信息
    """
    if not thinking_process or not isinstance(thinking_process, str):
        return []
    
    # 去除<think>和</think>标签
    if thinking_process.startswith("<think>"):
        thinking_process = thinking_process[7:]
    if thinking_process.endswith("</think>"):
        thinking_process = thinking_process[:-8]
    
    # 分析思考过程结构
    lines = thinking_process.split('\n')
    
    # 检查是否有迭代标记
    has_iteration_marker = False
    for line in lines:
        if re.search(r'开始第\d+轮迭代', line) or re.search(r'> \d+\.', line):
            has_iteration_marker = True
            break
    
    if not has_iteration_marker:
        # 查找Final Information部分
        final_info_idx = -1
        for i, line in enumerate(lines):
            if "Final Information" in line:
                final_info_idx = i
                break
        
        # 提取useful_info
        useful_info = None
        if final_info_idx >= 0 and final_info_idx + 1 < len(lines):
            useful_info_lines = []
            for i in range(final_info_idx + 1, min(final_info_idx + 5, len(lines))):
                if lines[i].strip():
                    useful_info_lines.append(lines[i])
            
            if useful_info_lines:
                useful_info = "\n".join(useful_info_lines)
        
        # 提取查询
        queries = []
        for line in lines:
            if line.strip().startswith(">") and "?" in line:
                query = line.strip()[1:].strip()
                if query and query not in queries:
                    queries.append(query)
        
        # 创建一个基本迭代
        return [{
            "round": 1,
            "content": lines,
            "queries": queries if queries else ["原始查询"],
            "useful_info": useful_info or "从思考过程中提取的信息"
        }]
    
    # 尝试识别迭代轮次
    iterations = []
    current_iteration = {"round": 1, "content": [], "queries": []}
    current_query = None
    
    for line in lines:
        # 检测新的迭代轮次
        round_match = re.search(r'开始第(\d+)轮迭代', line)
        if round_match:
            # 保存前一轮
            if current_iteration["content"]:
                iterations.append(current_iteration)
            
            # 开始新一轮
            round_num = int(round_match.group(1))
            current_iteration = {"round": round_num, "content": [line], "queries": []}
            continue
        
        # 检测查询行（以 "> 1. "等格式开头的行）
        query_match = re.search(r'> (\d+)\. (.+)', line)
        if query_match:
            query_text = query_match.group(2).strip()
            if query_text:
                current_iteration["queries"].append(query_text)
                current_query = query_text
            current_iteration["content"].append(line)
            continue
        
        # 检测发现有用信息的行
        if "发现有用信息" in line:
            info_match = re.search(r'发现有用信息:\s*(.+)', line)
            if info_match:
                info = info_match.group(1).strip()
                current_iteration["useful_info"] = info
            current_iteration["content"].append(line)
            continue
        
        # 检测Final Information部分
        if "Final Information" in line and current_iteration and "useful_info" not in current_iteration:
            info_idx = lines.index(line)
            if info_idx + 1 < len(lines):
                current_iteration["useful_info"] = lines[info_idx + 1]
        
        # 其他行
        current_iteration["content"].append(line)
    
    # 添加最后一轮
    if current_iteration["content"]:
        iterations.append(current_iteration)
    
    # 如果未能找到有效的迭代，创建一个基本迭代
    if not iterations:
        return [{
            "round": 1,
            "content": lines,
            "queries": ["原始查询"],
            "useful_info": "从思考过程中提取的信息"
        }]
    
    return iterations

async def process_feedback(message_id: str, query: str, is_positive: bool, thread_id: str, agent_type: str = "graph_agent") -> Dict:
    """
    处理用户对回答的反馈
    
    Args:
        message_id: 消息ID
        query: 查询内容
        is_positive: 是否为正面反馈
        thread_id: 线程ID
        agent_type: Agent类型
        
    Returns:
        Dict: 反馈处理结果
    """
    try:
        # 生成锁的键
        lock_key = f"{thread_id}_{query}"
        
        # 获取锁，防止并发处理同一个查询
        with feedback_manager.get_lock(lock_key):
            # 确保agent_type存在
            try:
                selected_agent = agent_manager.get_agent(agent_type)
            except ValueError:
                agent_type = "graph_agent"  # 回退到默认agent
                print(f"未知的agent类型，使用默认值: {agent_type}")
                selected_agent = agent_manager.get_agent(agent_type)
            
            # 根据反馈进行处理
            if is_positive:
                # 标记为高质量回答
                selected_agent.mark_answer_quality(query, True, thread_id)
                action = "缓存已被标记为高质量"
            else:
                # 负面反馈 - 从缓存中移除该回答
                selected_agent.clear_cache_for_query(query, thread_id)
                
                # 同时清除全局缓存
                if hasattr(selected_agent, 'global_cache_manager'):
                    # 直接使用原始查询作为键
                    selected_agent.global_cache_manager.delete(query)
                    action = "会话缓存和全局缓存已被清除"
                else:
                    action = "会话缓存已被清除，但无法访问全局缓存"
                
            # 更新操作时间戳
            feedback_manager.update_timestamp(lock_key)
            
            # 清理过期的锁
            feedback_manager.cleanup_expired_locks()
            
            return {
                "status": "success",
                "action": action
            }
    except Exception as e:
        print(f"处理反馈时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))