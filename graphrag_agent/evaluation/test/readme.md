可以通过--save_dir指定保存的评估结果的目录

**评估Graph Agent（带调试）：**

```bash
python evaluate_graph_agent.py --questions_file questions.json --verbose
```

**评估Hybrid Agent（带标准答案）：**

```bash
python evaluate_hybrid_agent.py --questions_file questions.json --golden_answers_file answer.json
```

**评估Fusion Agent：**

```bash
python evaluate_fusion_agent.py --questions_file questions.json --golden_answers_file answer.json
```

**仅评估Naive Agent的答案质量：**

```bash
python evaluate_naive_agent.py --questions_file questions.json --golden_answers_file answer.json --eval_type answer
```

**评估Deep Research Agent（使用增强版工具）：**

```bash
python evaluate_deep_agent.py --questions_file questions.json --use_deeper
```

### 比较所有Agent

使用主脚本来比较所有Agent的性能：

```bash
python evaluate_all_agents.py --questions_file questions.json --golden_answers_file answer.json  --verbose
```

**仅比较部分Agent：**

```bash
python evaluate_all_agents.py --questions_file questions.json --agents graph,hybrid,fusion
```

**仅比较检索性能：**

```bash
python evaluate_all_agents.py --questions_file questions.json --eval_type retrieval
```

**使用自定义指标：**

```bash
python evaluate_all_agents.py --questions_file questions.json --metrics em,f1,retrieval_precision
```

某次运行的结果（学校政务数据集）：

| 指标 | naive | hybrid | graph | deep | fusion |
| --- | --- | --- | --- | --- | --- |
| **答案质量指标** |  |  |  |  |  |
| answer_comprehensiveness | 0.9000 | 0.3000 | 1.0000 | 0.9333 | 0.9667 |
| em | 0.2667 | 0.2667 | 0.5667 | 0.7333 | 0.6000 |
| f1 | 0.4333 | 0.3091 | 0.5667 | 0.7500 | 0.6000 |
| factual_consistency | 0.9667 | 0.3000 | 0.9667 | 0.9333 | 0.9333 |
| response_coherence | 1.0000 | 0.5000 | 1.0000 | 1.0000 | 1.0000 |
| **LLM评估指标** |  |  |  |  |  |
| Comprehensiveness | 0.8667 | 0.3667 | 0.9000 | 0.9333 | 0.9667 |
| Directness | 0.9000 | 0.3333 | 0.9667 | 0.7333 | 0.8667 |
| Empowerment | 0.7667 | 0.3000 | 0.8000 | 0.8000 | 0.9000 |
| Relativeness | 0.9333 | 0.4000 | 0.9667 | 0.8667 | 0.9667 |
| Total | 0.8650 | 0.3517 | 0.9050 | 0.8433 | 0.9300 |
| **检索性能指标** |  |  |  |  |  |
| retrieval_latency | 12.4365 | 18.0870 | 9.0602 | 87.8841 | 180.8547 |
| retrieval_precision | 0.5667 | 0.5333 | 0.3667 | 0.7000 | 0.6000 |
| retrieval_utilization | 0.5667 | 0.3000 | 0.4000 | 0.4000 | 0.4667 |

[Lihua-World数据集](https://github.com/HKUDS/MiniRAG/tree/main/dataset/LiHua-World)运行结果（仅前三个agent，deepsearch太耗时且贵就没测）：

运行测试前，应该先把数据集放在files/下构建图谱，过程较久。

| 指标 | naive | hybrid | graph |
| --- | --- | --- | --- |
| **答案质量指标** |  |  |  |
| answer_comprehensiveness | 0.3920 | 0.6137 | 0.7488 |
| em | 0.1038 | 0.1187 | 0.2359 |
| f1 | 0.1064 | 0.1843 | 0.2813 |
| factual_consistency | 0.8758 | 0.8357 | 0.8772 |
| response_coherence | 0.6804 | 0.8622 | 0.8542 |
| **LLM评估指标** |  |  |  |
| Comprehensiveness | 0.3242 | 0.5936 | 0.6674 |
| Directness | 0.6708 | 0.6106 | 0.8501 |
| Empowerment | 0.2756 | 0.5203 | 0.6133 |
| Relativeness | 0.5790 | 0.6648 | 0.8442 |
| Total | 0.4451 | 0.5965 | 0.7346 |