import asyncio
import time
from datetime import datetime

from graphrag_agent.agents.deep_research_agent import DeepResearchAgent
from graphrag_agent.agents.naive_rag_agent import NaiveRagAgent
from graphrag_agent.agents.graph_agent import GraphAgent
from graphrag_agent.agents.hybrid_agent import HybridAgent
from graphrag_agent.agents.fusion_agent import FusionGraphRAGAgent


TEST_CONFIG = {
    "queries": [
        "优秀学生的申请条件是什么？",
        "学业奖学金有多少钱？",
        "大学英语考试的标准是什么？",
        "小明同学旷课了30学时，又私藏了吹风机，他还殴打了同学，他还能评选国家奖学金吗？",
    ],
    "max_wait_time": 300  # 每次测试的最大等待时间(秒)
}

async def test_agent_stream(agent, agent_name, query, thread_id, show_thinking=False, max_time=None):
    """测试特定Agent的流式响应"""
    if max_time is None:
        max_time = TEST_CONFIG["max_wait_time"]
        
    print(f"\n[测试] {agent_name} - 流式 - 查询: '{query}'")
    
    try:
        if not hasattr(agent, 'ask_stream'):
            print(f"[错误] {agent_name} 不支持流式输出")
            return {
                "agent": agent_name,
                "query": query,
                "error": "不支持流式输出",
                "success": False
            }
        
        # 记录性能指标
        start_time = time.time()
        chunk_count = 0
        total_chars = 0
        first_token_time = None
        collected_text = []
        
        # 设置超时
        timeout = start_time + max_time
        
        # 打印流式测试开始提示
        print(f"开始接收流式输出 (最长等待 {max_time} 秒)...")
        
        # 执行流式查询
        async for chunk in agent.ask_stream(query, thread_id=thread_id, **({"show_thinking": show_thinking} if agent_name.startswith("DeepResearchAgent") else {})):
            # 记录第一个token的时间
            if chunk_count == 0:
                first_token_time = time.time()
            
            chunk_count += 1
            
            # 将字典类型的chunk转换为字符串
            if isinstance(chunk, dict):
                if 'answer' in chunk:
                    chunk_text = chunk['answer']
                    print("\n[接收到最终答案字典]")
                else:
                    chunk_text = str(chunk)
                    print("\n[接收到中间结果字典]")
            else:
                chunk_text = str(chunk)
            
            # 追加到收集的文本
            collected_text.append(chunk_text)
            total_chars += len(chunk_text)
            
            # 每隔一定块数显示进度
            if chunk_count % 20 == 0:
                elapsed = time.time() - start_time
                print(f"已接收 {chunk_count} 块，共 {total_chars} 字符，耗时 {elapsed:.2f} 秒")
            
            # 超出最大等待时间则中断
            if time.time() > timeout:
                print(f"达到最大等待时间 {max_time} 秒，提前结束接收")
                break
        
        # 计算性能指标
        end_time = time.time()
        total_time = end_time - start_time
        time_to_first_token = (first_token_time - start_time) if first_token_time else None
        
        # 合并所有收集到的文本
        full_text = "".join(collected_text)
        
        # 显示测试结果
        print(f"\n[完成] 流式查询完成")
        print(f"- 总耗时: {total_time:.2f}秒")
        if time_to_first_token:
            print(f"- 首块延迟: {time_to_first_token:.2f}秒")
        print(f"- 数据块数: {chunk_count}个")
        print(f"- 总字符数: {total_chars}字符")
        
        # 显示结果预览
        if len(full_text) > 300:
            preview_text = full_text[:300] + "..."
        else:
            preview_text = full_text
        
        print(f"\n结果:\n{full_text}\n")  # 过长可以用preview_text
        
        return {
            "agent": agent_name,
            "query": query,
            "total_time": total_time,
            "time_to_first_token": time_to_first_token,
            "chunk_count": chunk_count,
            "total_chars": total_chars,
            "success": True
        }
    
    except Exception as e:
        print(f"[错误] {agent_name} 流式处理查询时出错: {str(e)}")
        return {
            "agent": agent_name,
            "query": query,
            "error": str(e),
            "success": False
        }

async def run_stream_tests():
    """运行所有流式测试"""
    print("\n===== 开始流式Agent测试 =====\n")
    
    # 创建所有agent实例
    agents = [
        # {"name": "DeepResearchAgent", "instance": DeepResearchAgent(use_deeper_tool=True)},
        # {"name": "NaiveRagAgent", "instance": NaiveRagAgent()},
        # {"name": "GraphAgent", "instance": GraphAgent()},
        # {"name": "HybridAgent", "instance": HybridAgent()},
        {"name": "FusionGraphRAGAgent", "instance": FusionGraphRAGAgent()}
    ]
    
    # 测试结果
    results = []
    
    # 遍历所有测试查询
    for query in TEST_CONFIG["queries"]:
        print(f"\n===== 测试查询: {query} =====")
        
        for agent_info in agents:
            agent_name = agent_info["name"]
            agent = agent_info["instance"]
            
            # 为每个测试创建唯一的线程ID
            thread_id = f"stream_{agent_name}_{int(time.time())}"
            
            # 执行测试
            result = await test_agent_stream(agent, agent_name, query, thread_id)
            results.append(result)
            
            # 只有DeepResearchAgent支持思考过程测试
            if agent_name == "DeepResearchAgent":
                print("\n--- 测试思考过程流式输出 ---")
                thinking_result = await test_agent_stream(
                    agent, f"{agent_name}(思考模式)", query, 
                    f"{thread_id}_thinking", show_thinking=True
                )
                results.append(thinking_result)
    
    # 打印测试总结
    successful_tests = sum(1 for r in results if r.get("success", False))
    total_tests = len(results)
    
    print("\n===== 测试总结 =====")
    print(f"成功测试: {successful_tests}/{total_tests}")
    
    # 计算平均指标
    valid_results = [r for r in results if r.get("success", False)]
    if valid_results:
        avg_total_time = sum(r.get("total_time", 0) for r in valid_results) / len(valid_results)
        avg_first_token = sum(r.get("time_to_first_token", 0) for r in valid_results if r.get("time_to_first_token")) / \
                         sum(1 for r in valid_results if r.get("time_to_first_token"))
        avg_chunks = sum(r.get("chunk_count", 0) for r in valid_results) / len(valid_results)
        
        print(f"平均总耗时: {avg_total_time:.2f}秒")
        print(f"平均首块延迟: {avg_first_token:.2f}秒")
        print(f"平均数据块数: {avg_chunks:.1f}个")
    
    # 显示失败的测试
    failed = [r for r in results if not r.get("success", False)]
    if failed:
        print("失败的测试:")
        for f in failed:
            agent = f.get("agent", "未知")
            query = f.get("query", "未知")
            error = f.get("error", "未知错误")
            print(f"- {agent} 处理 '{query}' 时失败: {error}")

if __name__ == "__main__":
    print(f"开始测试: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    asyncio.run(run_stream_tests())
    print(f"测试完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")