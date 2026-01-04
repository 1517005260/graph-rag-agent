# 知识图谱清理和重建指南

## 🎯 适用场景

以下情况需要清空并重建知识图谱：

✅ **必须重建**：
- 实体抽取逻辑重大更新（如本次生产级重构）
- 切换配置模式（传统 → 动态 或 反之）
- 修改领域或桥接点定义
- 实体/关系类型定义变化
- Prompt 模板大幅修改

⚠️ **建议重建**：
- Neo4j 数据损坏
- 图谱质量严重下降
- 缓存过期或损坏

❌ **无需重建**：
- 只改了搜索逻辑
- 只改了 Agent 推理逻辑
- 只改了前端 UI
- 只改了性能参数

---

## 🚀 快速开始（推荐）

### 一键清理和重建

```bash
# 1. 进入项目根目录
cd /home/user/graph-rag-agent

# 2. 执行清理脚本
./scripts/clean_and_rebuild.sh

# 3. 等待清理完成后，重新构建图谱
python graphrag_agent/integrations/build/main.py
```

---

## 📋 手动操作步骤

如果你想手动控制每一步：

### 步骤 1: 清空 Neo4j 数据库

**方式 A：通过 Docker（推荐，最彻底）**

```bash
# 停止容器并删除数据卷
docker compose down -v

# 重新启动
docker compose up -d

# 等待启动完成
sleep 20
```

**方式 B：通过 Cypher 命令**

打开 Neo4j Browser (http://localhost:7474)，执行：

```cypher
// 1. 删除所有节点和关系
MATCH (n) DETACH DELETE n;

// 2. 删除向量索引
DROP INDEX chunk_embedding_index IF EXISTS;
DROP INDEX entity_embedding_index IF EXISTS;

// 3. 查看并删除约束
SHOW CONSTRAINTS;
// 根据结果删除，例如：
// DROP CONSTRAINT unique_entity_id IF EXISTS;
```

### 步骤 2: 清空缓存

```bash
# 清空所有缓存目录
rm -rf cache/
rm -rf cache/graph/
rm -rf cache/global/

# 清空文件注册表
rm -f file_registry.json
```

### 步骤 3: 清理 Python 缓存（可选）

```bash
# 清理 __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 清理 .pyc 文件
find . -type f -name "*.pyc" -delete 2>/dev/null || true
```

### 步骤 4: 重新构建图谱

**方式一：完整构建（推荐）**

```bash
# 确保文档在 files/ 目录
ls files/

# 运行完整构建
python graphrag_agent/integrations/build/main.py
```

**方式二：V2 增量构建（L0+L1）**

```bash
# 完整流程（L0 快速通道 + L1 图谱构建）
python graphrag_agent/integrations/build/incremental_update_v2.py --mode full
```

---

## 🔍 验证重建是否成功

### 1. 检查 Neo4j

打开 Neo4j Browser (http://localhost:7474)：

```cypher
// 查看节点数量
MATCH (n) RETURN labels(n), count(*);

// 查看关系数量
MATCH ()-[r]->() RETURN type(r), count(*);

// 查看向量索引
SHOW INDEXES;
```

预期结果（根据你的文档量）：
- `__Entity__` 节点：应该比之前少 ~76%（质量控制效果）
- `__Chunk__` 节点：应该减少（GraphChunker 效果）
- `__Community__` 节点：社区数量
- 关系数量：应该比之前少 ~79%

### 2. 检查缓存

```bash
# 查看缓存目录（应该为空或只有新生成的缓存）
ls -la cache/
ls -la cache/graph/
ls -la cache/global/
```

### 3. 测试搜索

```bash
# 运行测试脚本
cd test/
python search_without_stream.py
```

---

## 📊 预期的改进效果

基于生产级重构，你应该看到：

| 指标 | 重建前 | 重建后 | 改进 |
|------|--------|--------|------|
| 实体数量 | 2544 | ~600 | -76% ⬇️ |
| 关系数量 | 11832 | ~2500 | -79% ⬇️ |
| LLM 调用次数 | ~2000 | ~500 | -75% ⬇️ |
| 构建时间 | T | ~0.4T | -60% ⬆️ |
| 实体质量 | 中等 | 高 | ⬆️ |
| Schema 一致性 | 低 | 高 | ⬆️ |

---

## ⚠️ 注意事项

1. **备份重要数据**（如果需要）
   ```bash
   # 备份配置
   cp .env .env.backup
   cp graph_config.json graph_config.json.backup 2>/dev/null || true

   # 备份文档（如果不在 Git 中）
   cp -r files/ files_backup/
   ```

2. **清理前确认**
   - 确保后端和前端服务已停止
   - 确保没有正在运行的构建任务
   - 确认你想要清空所有数据

3. **构建时间预估**
   - 小型项目（<10 文件）：5-10 分钟
   - 中型项目（10-50 文件）：20-60 分钟
   - 大型项目（50+ 文件）：1-3 小时

4. **GraphConfig 模式**
   - 如果使用动态配置，确保 `graph_config.json` 存在
   - 如果使用传统模式，确保 `settings.py` 配置正确

---

## 🐛 常见问题

### Q1: 清理后 Neo4j 无法启动

```bash
# 检查 Docker 日志
docker compose logs neo4j

# 重新启动
docker compose restart neo4j
```

### Q2: 重建时提示缓存错误

```bash
# 再次清空缓存
rm -rf cache/
mkdir -p cache/graph cache/global
```

### Q3: file_registry.json 删除后重建很慢

这是正常的，因为系统会重新处理所有文件。这也确保了使用最新的代码逻辑。

### Q4: 构建完成但图谱为空

检查：
1. `files/` 目录中是否有文档
2. `.env` 中的 LLM API 配置是否正确
3. 查看构建日志中是否有错误

---

## 📞 需要帮助？

如果遇到问题，请检查：
1. `server/logs/` 中的日志文件
2. 终端输出的错误信息
3. Neo4j Browser 中的数据状态

或提交 Issue：https://github.com/1517005260/graph-rag-agent/issues
