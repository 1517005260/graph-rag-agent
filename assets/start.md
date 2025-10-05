# 快速开始指南

## One-API 部署

使用 Docker 启动 One-API：

```bash
docker run --name one-api -d --restart always \
  -p 13000:3000 \
  -e TZ=Asia/Shanghai \
  -v /home/ubuntu/data/one-api:/data \
  justsong/one-api
```

在 One-API 控制台中配置第三方 API Key。本项目的所有 API 请求将通过 One-API 转发。

该项目的官方地址：https://github.com/songquanpeng/one-api

具体的填写方式可以看[这里](https://github.com/1517005260/graph-rag-agent/issues/7#issuecomment-2906770240)

**注意**：默认用管理员账号登录，用户名root，密码123456，进去之后可以改密码


## Neo4j 启动

```bash
cd graph-rag-agent/
docker compose up -d
```

默认账号密码：

```
用户名：neo4j
密码：12345678
```

## 环境搭建

```bash
conda create -n graphrag python==3.10
conda activate graphrag
cd graph-rag-agent/
pip install -r requirements.txt
```

注意：如需处理 `.doc` 格式（旧版 Word 文件），请根据操作系统安装相应依赖，详见 `requirements.txt` 中注释：

```txt
# Linux
sudo apt-get install python-dev-is-python3 libxml2-dev libxslt1-dev antiword unrtf poppler-utils

# Windows
pywin32>=302

textract==1.6.3  # Windows 无需安装
```

## .env 配置

在项目根目录下创建 `.env` 文件，示例如下：

```env
OPENAI_API_KEY = 'sk-xxx'  # api-key
OPENAI_BASE_URL = 'http://localhost:13000/v1' # url

OPENAI_EMBEDDINGS_MODEL = 'text-embedding-3-large'  # 向量嵌入模型
OPENAI_LLM_MODEL = 'gpt-4o'  # 对话模型

TEMPERATURE = 0   # 模型发散度，0-1，越大回答越天马行空
MAX_TOKENS = 2000  # 最大token

VERBOSE = True  # 调试模式

# neo4j 配置
NEO4J_URI='neo4j://localhost:7687'
NEO4J_USERNAME='neo4j'
NEO4J_PASSWORD='12345678'

# 缓存向量相似度匹配配置
# 可选值: 'openai' (复用RAG的向量模型), 'sentence_transformer' (使用本地模型)
CACHE_EMBEDDING_PROVIDER = 'openai'
# 当使用sentence_transformer时的模型名，模型会缓存到 ./cache/model 目录
CACHE_SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
# 模型缓存配置
MODEL_CACHE_ROOT = './cache'  # 缓存根目录，模型会保存到 {MODEL_CACHE_ROOT}/model

# LangSmith 配置（可选，若不需要此监控，可以直接注释）
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="xxx"
LANGSMITH_PROJECT="xxx"
```

**注意**：全流程测试通过的只有deepseek（20241226版本）以及gpt-4o，剩下的模型，比如deepseek（20250324版本）幻觉问题比较严重，有概率不遵循提示词，导致抽取实体失败；Qwen的模型可以抽取实体，但是好像不支持langchain/langgraph，所以问答的时候有概率报错，他们有自己的agent实现[Qwen-Agent](https://qwen.readthedocs.io/zh-cn/latest/framework/qwen_agent.html)

## 项目初始化

```bash
pip install -e .
```

## 知识图谱原始文件放置

请将原始文件放入 `files/` 文件夹，支持有目录的存放。当前支持以下格式（采用简单分块，后续会优化处理方式）：

```
- TXT（纯文本）
- PDF（PDF 文档）
- MD（Markdown）
- DOCX（新版 Word 文档）
- DOC（旧版 Word 文档）
- CSV（表格）
- JSON（结构化文本）
- YAML/YML（配置文件）
```

## 知识图谱配置（`config/settings.py`）

```python
# 基础设置
theme = "悟空传"
entity_types = ["人物", "妖怪", "位置"]
relationship_types = ["师徒", "师兄弟", "对抗", "对话", "态度", "故事地点", "其它"]

# 增量更新设置：冲突解决策略（新文件和手动编辑neo4j之间的冲突），可以是 "manual_first"（优先保留手动编辑），"auto_first"（优先自动更新）或 "merge"（尝试合并）
conflict_strategy="manual_first"

# 图谱参数
similarity_threshold = 0.9
community_algorithm = 'leiden'  # 可选：sllpa 或 leiden

# 文本分块参数
CHUNK_SIZE = 300
OVERLAP = 50
MAX_TEXT_LENGTH = 500000 # 最大处理段长

# 回答方式
response_type = "多个段落"

# Agent 工具描述
lc_description = "用于需要具体细节的查询，例如《悟空传》中的对话、场景描写等。"
gl_description = "用于宏观总结和分析，如人物关系、主题发展等。"
naive_description = "基础检索工具，返回最相关的原文段落。"

# 性能优化参数
# 并行处理配置
MAX_WORKERS = 4                # 并行工作线程数
BATCH_SIZE = 100               # 批处理大小
ENTITY_BATCH_SIZE = 50         # 实体处理批次大小
CHUNK_BATCH_SIZE = 100         # 文本块处理批次大小
EMBEDDING_BATCH_SIZE = 64      # 嵌入向量计算批次大小
LLM_BATCH_SIZE = 5             # LLM处理批次大小

# GDS相关配置
GDS_MEMORY_LIMIT = 6           # GDS内存限制(GB)
GDS_CONCURRENCY = 4            # GDS并发度
GDS_NODE_COUNT_LIMIT = 50000   # GDS节点数量限制
GDS_TIMEOUT_SECONDS = 300      # GDS超时时间(秒)

# 索引和社区检测配置
COMMUNITY_BATCH_SIZE = 50      # 社区处理批次大小

# 实体消歧和对齐配置
DISAMBIG_STRING_THRESHOLD = 0.7  # 字符串匹配阈值
DISAMBIG_VECTOR_THRESHOLD = 0.85  # 向量相似度阈值
DISAMBIG_NIL_THRESHOLD = 0.6  # NIL检测阈值
DISAMBIG_TOP_K = 5  # 候选实体数量

ALIGNMENT_CONFLICT_THRESHOLD = 0.5  # 冲突检测阈值
ALIGNMENT_MIN_GROUP_SIZE = 2  # 最小分组大小
```

## 构建知识图谱

```bash
cd graph-rag-agent/

# 初始全量构建
python build/main.py

# 单次变量（增量、减量）构建：
python build/incremental_update.py --once

# 后台守护进程，定期变量更新：
python build/incremental_update.py --daemon
```

**注意：** `main.py`是构建的全流程，如果需要单独跑某个流程，请先完成实体索引的构建，再进行 chunk 索引构建，否则会报错（chunk 索引依赖实体索引）。

## 知识图谱搜索测试

```bash
cd graph-rag-agent/test

# 查询前可以注释掉不想测试的Agent，防止运行过慢

# 非流式查询
python search_without_stream.py

# 流式查询
python search_with_stream.py
```

## 知识图谱评估

```bash
cd evaluator/test
# 查看对应 README 获取更多信息
```

## 示例问题配置（用于前端展示）

编辑 `config/settings.py` 中的 `examples` 字段：

```python
examples = [
    "《悟空传》的主要人物有哪些？",
    "唐僧和会说话的树讨论了什么？",
    "孙悟空跟女妖之间有什么故事？",
    "他最后的选择是什么？"
]
```

## 并发进程配置（`server/main.py`）

```python
# FastAPI 的并发进程数设置
workers = 2
```

## 深度搜索优化（建议禁用前端超时）

如需开启深度搜索功能，建议禁用前端超时限制，修改 `frontend/utils/api.py`：

```python
response = requests.post(
    f"{API_URL}/chat",
    json={
        "message": message,
        "session_id": st.session_state.session_id,
        "debug": st.session_state.debug_mode,
        "agent_type": st.session_state.agent_type
    },
    # timeout=120  # 建议注释掉此行
)
```

## 中文字体支持（Linux）

如需中文图表显示，可参考[字体安装教程](https://zhuanlan.zhihu.com/p/571610437)。默认使用英文绘图（`matplotlib`）。


## 启动前后端服务

```bash
# 启动后端
cd graph-rag-agent/
python server/main.py

# 启动前端
cd graph-rag-agent/
streamlit run frontend/app.py
```

**注意**：由于langchain版本问题，目前的流式是伪流式实现，即先完整生成答案，再分段返回。
