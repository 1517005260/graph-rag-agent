# 数据集下载

```
cd datasets/

# 下载HotpotQA dev集（用于测试，约44MB）
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

## 多智能体模块 HotpotQA 评测脚本用法

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