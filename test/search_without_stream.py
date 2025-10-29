import time
from datetime import datetime

from graphrag_agent.agents.deep_research_agent import DeepResearchAgent
from graphrag_agent.agents.naive_rag_agent import NaiveRagAgent
from graphrag_agent.agents.graph_agent import GraphAgent
from graphrag_agent.agents.hybrid_agent import HybridAgent
from graphrag_agent.agents.fusion_agent import FusionGraphRAGAgent

TEST_CONFIG = {
    "queries": [
        # "优秀学生的申请条件是什么？",
        "华东理工大学有哪些和国外学校交换项目？",
        "高校学生资助政策体系是什么？",
        "平安华理微信公众号二维码是什么？",
        # "学业奖学金有多少钱？",
        # "大学英语考试的标准是什么？",
        # "小明同学旷课了30学时，又私藏了吹风机，他还殴打了同学，他还能评选国家奖学金吗？",
    ]
}

def test_agent(agent, agent_name, query, thread_id, show_thinking=False):
    """测试特定Agent的响应"""
    print(f"\n[测试] {agent_name} - 查询: '{query}'")
    
    try:
        start_time = time.time()
        
        if show_thinking and hasattr(agent, 'ask_with_thinking'):
            result = agent.ask_with_thinking(query, thread_id=thread_id)
            if isinstance(result, dict) and 'answer' in result:
                answer = result['answer']
                # 打印思考过程中的关键信息
                thinking_keys = [k for k in result.keys() if k != 'answer']
                print(f"思考过程包含以下信息: {', '.join(thinking_keys)}")
            else:
                answer = str(result)
        else:
            answer = agent.ask(query, thread_id=thread_id)
        
        execution_time = time.time() - start_time
        
        # 处理答案预览
        if len(answer) > 300:
            answer_preview = answer[:300] + "..."
        else:
            answer_preview = answer
        
        print(f"[完成] 用时 {execution_time:.2f}秒，结果长度 {len(answer)} 字符")
        print(f"\n结果:\n{answer}\n") # 过长可以用 answer_preview
        
        return {
            "agent": agent_name,
            "query": query,
            "execution_time": execution_time,
            "result_length": len(answer),
            "success": True
        }
    
    except Exception as e:
        print(f"[错误] {agent_name} 处理查询时出错: {str(e)}")
        return {
            "agent": agent_name,
            "query": query,
            "error": str(e),
            "success": False
        }

def run_tests():
    """运行所有非流式测试"""
    print("\n===== 开始非流式Agent测试 =====\n")
    
    # 创建所有agent实例
    agents = [
        # {"name": "DeepResearchAgent", "instance": DeepResearchAgent(use_deeper_tool=True)},
        {"name": "NaiveRagAgent", "instance": NaiveRagAgent()},
        # {"name": "GraphAgent", "instance": GraphAgent()},
        # {"name": "HybridAgent", "instance": HybridAgent()},
        # {"name": "FusionGraphRAGAgent", "instance": FusionGraphRAGAgent()}
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
            thread_id = f"test_{agent_name}_{int(time.time())}"
            
            # 执行测试
            result = test_agent(agent, agent_name, query, thread_id)
            results.append(result)
            
            # 只有DeepResearchAgent支持思考过程测试
            if agent_name == "DeepResearchAgent":
                print("\n--- 测试思考过程 ---")
                thinking_result = test_agent(agent, f"{agent_name}(思考模式)", query, f"{thread_id}_thinking", show_thinking=True)
                results.append(thinking_result)
    
    # 打印测试总结
    successful_tests = sum(1 for r in results if r.get("success", False))
    total_tests = len(results)
    
    print("\n===== 测试总结 =====")
    print(f"成功测试: {successful_tests}/{total_tests}")
    
    # 计算平均执行时间
    execution_times = [r.get("execution_time", 0) for r in results if "execution_time" in r]
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        print(f"平均执行时间: {avg_time:.2f}秒")
    
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
    run_tests()
    print(f"测试完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")