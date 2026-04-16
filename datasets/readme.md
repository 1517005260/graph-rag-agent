# 数据集下载

```
cd datasets/

# 下载HotpotQA dev集（用于测试，约44MB）
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

## 多智能体模块 HotpotQA 评测脚本用法

### 数据与入口索引

| 数据 / 入口 | 默认位置 | 作用 | 常见输出 |
|---|---|---|---|
| HotpotQA dev 集 | `datasets/hotpot_dev_distractor_v1.json` | 离线评测多智能体检索与回答质量 | `outputs/hotpot_predictions.json` |
| 流式问答测试 | `test/search_with_stream.py` | 观察实时输出、思考过程与调试信息 | 终端流式文本、执行轨迹 |
| 非流式问答测试 | `test/search_without_stream.py` | 对比不同 Agent 的最终回答 | 终端完整回答、结构化结果 |
| 图谱构建入口 | `graphrag_agent/integrations/build/main.py` | 从原始文档生成图谱、索引和社区 | Neo4j 图数据、缓存文件 |
| 前端演示入口 | `frontend/app.py` | 在网页中查看问答与知识图谱 | 浏览器交互界面 |

脚本路径：`test/hotpot_multi_agent_eval.py`

使用步骤：
1. 确保已下载 HotpotQA 数据集（默认路径为 `datasets/hotpot_dev_distractor_v1.json`）。
2. 配置所需的大模型环境变量（如 `OPENAI_API_KEY`）。
3. 在项目根目录执行示例命令：
   ```
   python test/hotpot_multi_agent_eval.py --limit 20 --retrieval-top-k 5 \
       --report-type short_answer --llm-eval --predictions outputs/hotpot_predictions.json
   ```
   主要参数说明：
   - `--dataset`: HotpotQA 数据集 JSON 路径（默认 `datasets/hotpot_dev_distractor_v1.json`）。
   - `--limit`: 仅评估前 N 条样本，缺省为全部。
   - `--retrieval-top-k`: TF-IDF 检索返回段落数。
   - `--report-type`: Reporter 输出类型，可选 `short_answer` 或 `long_document`。
   - `--llm-eval`: 启用 LLM 判定答案正确性。
   - `--predictions`: 保存预测与指标的输出文件路径。

## Environment Requirements

- Configure credentials via environment variables before startup.
- Refer to `.env.example` for the minimum required keys.

## Quick Start

- Install project dependencies according to this module.
- Run the module with your standard build command (for example `mvn test`, `npm run dev`, or equivalent script in this repo).
