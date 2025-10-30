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

### 或者

1. 直接使用第三方代理平台，如[云雾api](https://yunwu.ai/)等，使用方法同one-api，`.env`中api-key写中转站给你的key，url写中转站的url

2. **建议**：使用更先进的[new-api](https://github.com/QuantumNous/new-api)，使用方法基本同one-api

```bash
# 使用SQLite部署new-api
docker run --name new-api -d --restart always -p 13000:3000 -e TZ=Asia/Shanghai -v /home/ubuntu/data/new-api:/data calciumion/new-api:latest
```


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

## MinerU安装

见[这个文件](./mineru.md)

## MinerU 集成

### 目录与环境约定

- 项目根目录结构需保持：
  ```
  /path/to/workspace/
  ├── graph-rag-agent/        # 当前工程
  └── MinerU/                # 官方 MinerU 仓库
  ```
- Graphrag 项目使用 `conda activate graphrag`
- MinerU 服务使用 `conda activate mineru`

### MinerU 服务器启动

0. 推荐配置：存储20G以上，显存8G以上再使用MinerU进行解析，可以通过auto-dl平台，ssh隧道的功能映射到本地。如果配置不足，MinerU解析会很慢，不建议使用。

1. 在 `graph-rag-agent` 目录内运行：
   ```bash
   conda activate mineru
   cd ../graphrag-agent
   python mineru_server.py
   ```

2. MinerU 默认监听 `http://localhost:8899`，解析结果写入 `graph-rag-agent/mineru_outputs/`。

### Graphrag 侧配置

1. 在 `graph-rag-agent/.env` 中补充 MinerU 相关变量（已在 `.env.example` 提供）：
   ```env
   DOCUMENT_PROCESSOR_MODE = 'mineru'  # legacy | mineru
   MINERU_API_URL = 'http://localhost:8899'
   MINERU_HOME = '../MinerU'
   MINERU_OUTPUT_DIR = './mineru_outputs'
   MINERU_TEMP_DIR = './.tmp/mineru_temp'
   ```
2. 传统模式(`legacy`)：仅处理 `files/` 目录中历史支持的文本格式。
3. MinerU 模式(`mineru`)：将原始文件放入 `files/`，系统自动通过 MinerU 解析至 `mineru_outputs/`，并使用解析结果入库；图片分片会以占位符形式记录，前端可根据 `modal_segments` 中的 `image_relative_path` 加载。
4. 开启MinerU模式解析时，服务器必须启动，否则自动回退到legacy模式。
5. MinerU 的解析结果会缓存在 `cache/mineru_registry.json` 与 `cache/mineru_cache/` 下，同一文件在内容和解析参数均未变化时不会重复调用 MinerU。如果需要强制重解析，可删除对应缓存文件或清空上述目录。

## 环境变量配置 (.env)

### 配置说明

项目配置已统一至 `.env` 文件，除知识图谱实体/关系模式外，所有运行参数均可通过环境变量覆盖。请在项目根目录创建 `.env` 文件进行配置。

### 必选配置项

以下配置为必填项，项目无法运行时需首先检查这些配置（具体含义可见文件里的注释）：

```env
# ===== LLM 模型配置 =====
OPENAI_API_KEY = 'sk-xxx'
OPENAI_BASE_URL = 'http://localhost:13000/v1'
OPENAI_EMBEDDINGS_MODEL = 'text-embedding-3-large'
OPENAI_LLM_MODEL = 'gpt-4o'

# ===== Neo4j 数据库配置 =====
NEO4J_URI = 'neo4j://localhost:7687'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = '12345678'
```

### 推荐修改配置项

以下配置根据实际使用场景建议调整：

```env
# ===== 缓存向量模型配置 =====
# 推荐使用第三方embedding模型api，省事。以下是需要下载的配置
CACHE_EMBEDDING_PROVIDER = 'sentence_transformer'
CACHE_SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

# ===== 并发与性能配置 =====
# 根据机器性能调整，4核CPU推荐值如下
FASTAPI_WORKERS = 2
MAX_WORKERS = 4  # 如果one-api出现“上游负载等”的报错，可以把这个改成2
MA_WORKER_EXECUTION_MODE = 'parallel'
MA_WORKER_MAX_CONCURRENCY = 4
BATCH_SIZE = 100

# ===== GDS 内存配置 =====
# 根据服务器内存调整，单位 GB
GDS_MEMORY_LIMIT = 6

# ===== 文本处理参数 =====
# 根据文档特性调整分块大小
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
```

### 默认即可配置项

以下配置一般无需修改，保持默认值即可：

```env
# ===== LLM 生成参数 =====
TEMPERATURE = 0
MAX_TOKENS = 2000
VERBOSE = True

# ===== 文本处理 =====
MAX_TEXT_LENGTH = 500000
SIMILARITY_THRESHOLD = 0.9
RESPONSE_TYPE = '多个段落'

# ===== 批处理大小 =====
ENTITY_BATCH_SIZE = 50
CHUNK_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 64
LLM_BATCH_SIZE = 5
COMMUNITY_BATCH_SIZE = 50

# ===== GDS 运行参数 =====
GDS_CONCURRENCY = 4
GDS_NODE_COUNT_LIMIT = 50000
GDS_TIMEOUT_SECONDS = 300

# ===== 实体消歧配置 =====
DISAMBIG_STRING_THRESHOLD = 0.7
DISAMBIG_VECTOR_THRESHOLD = 0.85
DISAMBIG_NIL_THRESHOLD = 0.6
DISAMBIG_TOP_K = 5
ALIGNMENT_CONFLICT_THRESHOLD = 0.5
ALIGNMENT_MIN_GROUP_SIZE = 2

# ===== Neo4j 连接池 =====
NEO4J_MAX_POOL_SIZE = 10
NEO4J_REFRESH_SCHEMA = false

# ===== 缓存系统配置 =====
MODEL_CACHE_ROOT = './cache'
CACHE_ROOT = './cache'
CACHE_DIR = './cache'
CACHE_MEMORY_ONLY = false
CACHE_MAX_MEMORY_SIZE = 100
CACHE_MAX_DISK_SIZE = 1000
CACHE_THREAD_SAFE = true
CACHE_ENABLE_VECTOR_SIMILARITY = true
CACHE_SIMILARITY_THRESHOLD = 0.9
CACHE_MAX_VECTORS = 10000

# ===== 相似实体检测 =====
SIMILAR_ENTITY_WORD_EDIT_DISTANCE = 3
SIMILAR_ENTITY_BATCH_SIZE = 500
SIMILAR_ENTITY_MEMORY_LIMIT = 6
SIMILAR_ENTITY_TOP_K = 10

# ===== 搜索工具参数 =====
SEARCH_CACHE_MEMORY_SIZE = 200
SEARCH_VECTOR_LIMIT = 5
SEARCH_TEXT_LIMIT = 5
SEARCH_SEMANTIC_TOP_K = 5
SEARCH_RELEVANCE_TOP_K = 5
NAIVE_SEARCH_TOP_K = 3

LOCAL_SEARCH_TOP_CHUNKS = 3
LOCAL_SEARCH_TOP_COMMUNITIES = 3
LOCAL_SEARCH_TOP_OUTSIDE_RELS = 10
LOCAL_SEARCH_TOP_INSIDE_RELS = 10
LOCAL_SEARCH_TOP_ENTITIES = 10
LOCAL_SEARCH_INDEX_NAME = 'vector'

GLOBAL_SEARCH_LEVEL = 0
GLOBAL_SEARCH_BATCH_SIZE = 5

HYBRID_SEARCH_ENTITY_LIMIT = 15
HYBRID_SEARCH_MAX_HOP = 2
HYBRID_SEARCH_TOP_COMMUNITIES = 3
HYBRID_SEARCH_BATCH_SIZE = 10
HYBRID_SEARCH_COMMUNITY_LEVEL = 0

# ===== Server 运行参数 =====
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8000
SERVER_RELOAD = false
SERVER_LOG_LEVEL = 'info'
SERVER_WORKERS = 2

# ===== 前端运行参数 =====
FRONTEND_API_URL = 'http://localhost:8000'
FRONTEND_DEFAULT_AGENT = 'naive_rag_agent'
FRONTEND_DEFAULT_DEBUG = false
FRONTEND_SHOW_THINKING = true
FRONTEND_USE_DEEPER_TOOL = true
FRONTEND_USE_STREAM = true
FRONTEND_USE_CHAIN_EXPLORATION = true

# ===== 知识图谱可视化参数 =====
KG_PHYSICS_ENABLED = true
KG_NODE_SIZE = 25
KG_EDGE_WIDTH = 2
KG_SPRING_LENGTH = 150
KG_GRAVITY = -5000

# ===== Agent 参数 =====
AGENT_RECURSION_LIMIT = 5
AGENT_CHUNK_SIZE = 4
AGENT_STREAM_FLUSH_THRESHOLD = 40
DEEP_AGENT_STREAM_FLUSH_THRESHOLD = 80
FUSION_AGENT_STREAM_FLUSH_THRESHOLD = 60
```

### 可选配置项

以下配置为可选功能，不需要可以不配置或注释掉：

```env
# ===== LangSmith 监控（可选）=====
# 不需要可以完全注释掉此部分
LANGSMITH_TRACING = true
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = "xxx"
LANGSMITH_PROJECT = "xxx"
```

### 配置模板获取

完整配置模板请参考项目根目录的 `.env.example` 文件，可直接复制重命名为 `.env` 后修改。

### 模型兼容性说明

全流程测试通过的模型：
- DeepSeek (20241226版本)
- GPT-4o

已知问题模型：
- DeepSeek (20250324版本)：幻觉问题严重，可能导致实体抽取失败
- Qwen 系列：可以抽取实体，但与 LangChain/LangGraph 兼容性存在问题，建议使用其官方 [Qwen-Agent](https://qwen.readthedocs.io/zh-cn/latest/framework/qwen_agent.html) 框架

## 项目初始化

```bash
pip install -e .
```

## 知识图谱原始文件放置

请将原始文件放入 `files/` 文件夹，支持有目录的存放。`legacy`模式支持以下格式（采用简单分块）：

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

`mineru`模式采用ocr解析，统一解析为`markdown`文件，还支持以下文件：

```
MINERU_SUPPORTED_SUFFIXES = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".webp",
        ".jp2",
    }
```

## 知识图谱实体与关系配置

除环境变量配置外，知识图谱的实体类型和关系类型需要在代码中配置。

编辑 `graphrag_agent/config/settings.py`：

```python
# 知识图谱主题
theme = "华东理工大学学生管理"

# 实体类型定义
entity_types = [
    "学生类型",
    "奖学金类型",
    "处分类型",
    "部门",
    "学生职责",
    "管理规定",
]

# 关系类型定义
relationship_types = [
    "申请",
    "评定",
    "撤销",
    "负责",
    "担任",
    "管理",
    "权利义务",
    "互斥",
]

# 冲突解决策略（也可通过环境变量 GRAPH_CONFLICT_STRATEGY 覆盖）
# manual_first: 优先保留手动编辑
# auto_first: 优先自动更新
# merge: 尝试合并
conflict_strategy = "manual_first"

# 社区检测算法（也可通过环境变量 GRAPH_COMMUNITY_ALGORITHM 覆盖）
# leiden / sllpa （sllpa如果发现不了社区，建议换成leiden）
community_algorithm = "leiden"
```

## 构建知识图谱

```bash
cd graph-rag-agent/

# 如果采用MinerU解析，需要先 python mineru_server.py

# 初始全量构建
python graphrag_agent/integrations/build/main.py

# 单次变量（增量、减量）构建：
python graphrag_agent/integrations/build/incremental_update.py --once

# 后台守护进程，定期变量更新：
python graphrag_agent/integrations/build/incremental_update.py --daemon
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
cd graphrag_agent/evaluation/test
# 查看对应 README 获取更多信息
```

## 前端示例问题配置

编辑 `graphrag_agent/config/settings.py` 中的 `examples` 字段：

```python
examples = [
    "旷课多少学时会被退学？",
    "国家奖学金和国家励志奖学金互斥吗？",
    "优秀学生要怎么申请？",
    "那上海市奖学金呢？"
]
```

## 配置体系说明

项目配置分为三层：

1. `.env` 文件：所有运行时参数、密钥、性能调优参数
2. `graphrag_agent/config/settings.py`：知识图谱结构配置（实体/关系类型、示例问题等）
3. `server/server_config/settings.py` 和 `frontend/frontend_config/settings.py`：自动继承上层配置

大部分配置可通过 `.env` 直接控制，无需修改代码。

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
