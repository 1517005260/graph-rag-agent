"""
实体关系提取器（生产级重构版本）

整合了生产级 GraphRAG 验证的三板斧优化：
1. Graph 专用 Chunk（已在 chunker 层完成）
2. 强化版 Prompt（硬约束 + 白名单）
3. Post-process（标准化 + 去重 + 频率过滤）

关键改进：
- 实体/关系类型白名单
- JSON 格式输出（更可靠）
- 实体标准化和去重
- 频率过滤（≥2 次）
- Levenshtein 距离合并
- 关系合法性校验
"""

import time
import os
import pickle
import json
import re
import concurrent.futures
from typing import List, Tuple, Optional, Dict
from collections import Counter
from difflib import SequenceMatcher

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from graphrag_agent.graph.core import retry, generate_hash
from graphrag_agent.config.settings import MAX_WORKERS as DEFAULT_MAX_WORKERS, BATCH_SIZE as DEFAULT_BATCH_SIZE


# =========================
# 生产级配置（硬约束）
# =========================

ALLOWED_ENTITY_TYPES = {
    "POLICY",       # 制度、政策、办法、条例
    "PROCESS",      # 流程、步骤、阶段
    "CONDITION",    # 条件、资格、标准
    "ORGANIZATION", # 组织、机构、部门
    "DOCUMENT"      # 正式文件名称
}

ALLOWED_RELATION_TYPES = {
    "HAS_CONDITION",
    "HAS_STEP",
    "ISSUED_BY",
    "APPLIES_TO",
    "PART_OF",
    "REQUIRES"
}

MIN_ENTITY_FREQUENCY = 2            # 最小实体频率
NAME_SIMILARITY_THRESHOLD = 0.85    # 名称相似度阈值


# =========================
# 工具函数（生产级）
# =========================

def normalize_entity_name(name: str) -> str:
    """
    标准化实体名称

    规则：
    - 去除空格
    - 统一括号格式：全角 → 半角
    """
    if not name:
        return ""

    return (
        name.strip()
        .replace(" ", "")
        .replace("（", "(")
        .replace("）", ")")
        .replace("【", "[")
        .replace("】", "]")
    )


def is_similar(a: str, b: str) -> bool:
    """判断两个实体名称是否相似（基于 SequenceMatcher）"""
    return SequenceMatcher(None, a, b).ratio() >= NAME_SIMILARITY_THRESHOLD


def _extract_json_dict(text: str) -> Optional[Dict]:
    """
    从 LLM 输出中提取 JSON dict。兼容 ```json fence、前后解释文字。

    尝试策略：
    1. 去除 ```json ... ``` fence
    2. 截取第一个 {...}，避免前后有解释
    3. 解析为 dict
    """
    if not text:
        return None

    t = text.strip()

    # 去掉 ```json ... ``` / ``` ... ```
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)

    # 截取第一个 {...}，避免前后有解释
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    t = m.group(0)

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# =========================
# 后处理核心逻辑（生产级）
# =========================

def post_process_entities(raw_entities: List[Dict], allowed_types: set = None) -> List[Dict]:
    """
    实体后处理（生产级验证 + 动态 Schema）

    步骤：
    1. normalize：标准化名称
    2. type_filter：类型过滤（动态白名单）
    3. frequency_filter：频率过滤（≥2 次）
    4. deduplicate：去重（Levenshtein 距离）

    Args:
        raw_entities: 原始实体列表
        allowed_types: 允许的实体类型（动态 Schema，默认使用全局白名单）
    """
    if not raw_entities:
        return []

    # 使用动态 Schema 或默认白名单
    if allowed_types is None:
        allowed_types = ALLOWED_ENTITY_TYPES

    # 1. normalize
    for e in raw_entities:
        if "name" in e:
            e["name"] = normalize_entity_name(e.get("name", ""))

    # 2. type filter（动态白名单）
    entities = [
        e for e in raw_entities
        if e.get("type") in allowed_types and e.get("name")
    ]

    # 3. frequency filter（≥2 次）
    freq = Counter(e["name"] for e in entities)
    entities = [
        e for e in entities
        if freq[e["name"]] >= MIN_ENTITY_FREQUENCY
    ]

    # 4. deduplicate by similarity（Levenshtein 距离）
    deduped = []
    for e in entities:
        if not any(is_similar(e["name"], u["name"]) for u in deduped):
            deduped.append(e)

    print(f"✅ 实体后处理：{len(raw_entities)} → {len(deduped)} 个实体（Schema: {len(allowed_types)} 类型）")
    return deduped


def post_process_relations(
    raw_relations: List[Dict],
    entities: List[Dict],
    allowed_relation_types: set = None
) -> List[Dict]:
    """
    关系后处理（生产级验证 + 动态 Schema）

    步骤：
    1. 验证关系类型（动态白名单）
    2. 验证 source/target 实体存在
    3. 去重

    Args:
        raw_relations: 原始关系列表
        entities: 实体列表
        allowed_relation_types: 允许的关系类型（动态 Schema，默认使用全局白名单）
    """
    if not raw_relations or not entities:
        return []

    # 使用动态 Schema 或默认白名单
    if allowed_relation_types is None:
        allowed_relation_types = ALLOWED_RELATION_TYPES

    entity_names = {normalize_entity_name(e["name"]) for e in entities}
    cleaned = []
    seen = set()

    for r in raw_relations:
        src = normalize_entity_name(r.get("source", ""))
        tgt = normalize_entity_name(r.get("target", ""))
        r_type = r.get("type")

        # 验证（动态白名单）
        if (
            src in entity_names and
            tgt in entity_names and
            r_type in allowed_relation_types
        ):
            key = (src, tgt, r_type)
            if key not in seen:
                cleaned.append({
                    "source": src,
                    "target": tgt,
                    "type": r_type
                })
                seen.add(key)

    print(f"✅ 关系后处理：{len(raw_relations)} → {len(cleaned)} 条关系（Schema: {len(allowed_relation_types)} 类型）")
    return cleaned


# =========================
# 主类（生产级重构）
# =========================

class EntityRelationExtractor:
    """
    实体关系提取器（生产级版本 + Schema-aware Routing）

    特点：
    - 强化版 Prompt（硬约束 + 白名单）
    - JSON 格式输出（可靠解析）
    - 实体后处理（标准化 + 去重 + 频率过滤）
    - 关系后处理（合法性校验）
    - Schema-aware Routing（文件级 Domain 识别）
    - 动态 Schema（支持多领域）
    - 保留原有的缓存和并行处理逻辑
    """

    def __init__(self, llm, system_template, human_template,
                 entity_types: List[str], relationship_types: List[str],
                 cache_dir="./cache/graph", max_workers=4, batch_size=5,
                 graph_config=None):
        """
        初始化实体关系提取器（+ GraphConfig 支持 + 缓存版本隔离）

        Args:
            llm: 语言模型
            system_template: 系统提示模板（生产级）
            human_template: 用户提示模板
            entity_types: 实体类型列表（兼容性，实际使用白名单）
            relationship_types: 关系类型列表（兼容性，实际使用白名单）
            cache_dir: 缓存目录（支持版本隔离）
            max_workers: 并行工作线程数（支持 "auto" 或 0 表示自动计算）
            batch_size: 批处理大小
            graph_config: GraphConfig 实例（可选，用于 Schema-aware routing）
        """
        self.llm = llm
        self.entity_types = entity_types
        self.relationship_types = relationship_types
        self.chat_history = []

        # 🔥 新增：GraphConfig 支持
        self.graph_config = graph_config

        # 🔥 新增：保存模板用于缓存版本控制
        self.system_template = system_template
        self.human_template = human_template

        # 🔥 新增：提取 model_name 用于缓存隔离
        self.model_name = self._extract_model_name(llm)

        # 设置分隔符（兼容旧格式）
        self.tuple_delimiter = " : "
        self.record_delimiter = "\n"
        self.completion_delimiter = "\n\n"

        # 创建提示模板
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            MessagesPlaceholder("chat_history"),
            human_message_prompt
        ])

        # 创建处理链
        self.chain = self.chat_prompt | self.llm

        # 🔥 缓存设置（版本隔离）
        # 计算 prompt 版本 hash（混合 system 和 human template）
        self.prompt_version = generate_hash(system_template + human_template)[:8]

        # 构建带版本隔离的缓存目录
        # 格式: cache_dir/model_name/prompt_version/
        self.cache_dir = os.path.join(
            cache_dir,
            self.model_name,
            self.prompt_version
        )
        self.enable_cache = True

        # 确保缓存目录存在
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"创建缓存目录（版本隔离）: {self.cache_dir}")

        # 🔥 并行处理配置（支持动态线程数）
        self.max_workers = self._compute_max_workers(max_workers)
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE

        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"🔥 生产级实体提取器已初始化")
        print(f"   - Model: {self.model_name}")
        print(f"   - Prompt Version: {self.prompt_version}")
        print(f"   - Cache Dir: {self.cache_dir}")
        print(f"   - Max Workers: {self.max_workers}")
        print(f"   - 实体类型白名单：{ALLOWED_ENTITY_TYPES}")
        print(f"   - 关系类型白名单：{ALLOWED_RELATION_TYPES}")
        print(f"   - 最小实体频率：{MIN_ENTITY_FREQUENCY}")

    def _extract_model_name(self, llm) -> str:
        """
        从 LLM 对象中提取模型名称

        尝试策略：
        1. llm.model_name (LangChain ChatOpenAI)
        2. llm.model (某些 LLM 实现)
        3. llm.__class__.__name__ (fallback)

        Returns:
            str: 模型名称（用于缓存隔离）
        """
        # 策略 1: model_name 属性
        if hasattr(llm, 'model_name') and llm.model_name:
            return str(llm.model_name).replace('/', '_')  # 避免路径问题

        # 策略 2: model 属性
        if hasattr(llm, 'model') and llm.model:
            return str(llm.model).replace('/', '_')

        # 策略 3: 类名 fallback
        return llm.__class__.__name__

    def _compute_max_workers(self, max_workers) -> int:
        """
        计算实际的最大工作线程数

        支持：
        - 整数 N: 使用 N 个线程
        - "auto" 或 0: 自动计算（CPU 核数 + 4，最多 32）
        - None: 使用默认配置

        Returns:
            int: 实际线程数
        """
        # 如果是 "auto" 或 0，动态计算
        if max_workers == "auto" or max_workers == 0:
            # 使用 Python ThreadPoolExecutor 的默认策略
            # min(32, (cpu_count or 1) + 4)
            cpu_count = os.cpu_count() or 1
            computed = min(32, cpu_count + 4)
            print(f"动态计算线程数：CPU 核数 {cpu_count} → {computed} 个线程")
            return computed

        # 如果是 None，使用默认配置
        if max_workers is None:
            return DEFAULT_MAX_WORKERS

        # 如果是整数，直接使用
        if isinstance(max_workers, int) and max_workers > 0:
            return max_workers

        # 其他情况，fallback 到默认值
        print(f"⚠️ 无效的 max_workers 值: {max_workers}，使用默认值 {DEFAULT_MAX_WORKERS}")
        return DEFAULT_MAX_WORKERS

    def _generate_cache_key(self, text: str) -> str:
        """生成文本的缓存键"""
        return generate_hash(text)

    def _cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _save_to_cache(self, cache_key: str, result: str) -> None:
        """保存结果到缓存"""
        if not self.enable_cache:
            return

        cache_path = self._cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"缓存保存错误: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """从缓存加载结果"""
        if not self.enable_cache:
            return None

        cache_path = self._cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                    self.cache_hits += 1
                    return result
            except Exception as e:
                print(f"缓存加载错误: {e}")

        self.cache_misses += 1
        return None

    def _route_domain(self, filename: str, content: str) -> str:
        """
        路由文档领域（文件级 Domain 识别）

        策略：
        1. 如果有 GraphConfig，使用 GraphConfig.route_domain()
        2. 否则返回 "default"（使用全局白名单）

        Args:
            filename: 文件名
            content: 文件内容

        Returns:
            domain: 领域标识（如 "student_policy", "hr_policy", "default"）
        """
        if self.graph_config and hasattr(self.graph_config, 'route_domain'):
            return self.graph_config.route_domain(filename, content)
        return "default"

    def _get_schema(self, domain: str) -> Tuple[set, set]:
        """
        获取领域的 Schema（实体类型 + 关系类型）

        策略：
        1. 如果有 GraphConfig，使用 GraphConfig.get_schema(domain)
        2. 否则返回全局白名单

        Args:
            domain: 领域标识

        Returns:
            (entity_types, relation_types): 实体类型集合和关系类型集合
        """
        if self.graph_config and hasattr(self.graph_config, 'get_schema'):
            schema = self.graph_config.get_schema(domain)
            return (
                set(schema.get("entity_types", ALLOWED_ENTITY_TYPES)),
                set(schema.get("relation_types", ALLOWED_RELATION_TYPES))
            )
        return (ALLOWED_ENTITY_TYPES, ALLOWED_RELATION_TYPES)

    @retry(times=3, exceptions=(Exception,), delay=1.0)
    def _process_single_chunk(
        self,
        input_text: str,
        domain_entity_types: set = None,
        domain_relation_types: set = None
    ) -> Dict:
        """
        处理单个文本块（生产级重构 + Schema-aware）

        流程：
        1. 检查缓存
        2. 调用 LLM
        3. 解析 JSON
        4. 后处理实体和关系（使用 domain schema）
        5. 保存缓存
        6. 返回 dict 结果

        Args:
            input_text: 输入文本
            domain_entity_types: 领域实体类型（可选，默认使用全局白名单）
            domain_relation_types: 领域关系类型（可选，默认使用全局白名单）

        Returns:
            Dict: 包含 entities, relations 等字段的字典
        """
        # 使用默认 schema（如果未提供）
        if domain_entity_types is None:
            domain_entity_types = ALLOWED_ENTITY_TYPES
        if domain_relation_types is None:
            domain_relation_types = ALLOWED_RELATION_TYPES

        # 生成缓存键
        cache_key = self._generate_cache_key(input_text)

        # 尝试从缓存加载
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            # 确保缓存结果是dict
            if isinstance(cached_result, dict):
                return cached_result
            # ✅ 兼容旧的字符串缓存格式 - 尝试解析为 JSON
            if isinstance(cached_result, str):
                obj = _extract_json_dict(cached_result)
                if isinstance(obj, dict) and ("entities" in obj or "relationships" in obj or "relations" in obj):
                    # 成功解析出有效的实体/关系数据
                    return obj
                # 如果字符串无法解析为有效 JSON
                print(f"⚠️ 缓存字符串无法解析为有效JSON，返回空结果")
                return {"entities": [], "relations": [], "relationships": []}
            # 其他类型的缓存（不应该发生）
            print(f"⚠️ 缓存格式异常(type={type(cached_result)})，返回空结果")
            return {"entities": [], "relations": [], "relationships": []}

        # 未缓存，调用 LLM 处理
        response = self.chain.invoke({
            "chat_history": self.chat_history,
            "entity_types": self.entity_types,
            "relationship_types": self.relationship_types,
            "tuple_delimiter": self.tuple_delimiter,
            "record_delimiter": self.record_delimiter,
            "completion_delimiter": self.completion_delimiter,
            "input_text": input_text
        })

        # 从 AIMessage 获取内容
        raw = getattr(response, "content", response)  # 兼容 AIMessage / str

        # 🔥 生产级后处理（关键改进 + 动态 Schema）
        result = {
            "entities": [],
            "relations": [],
            "relationships": [],
            "domains": [],
            "bridges": []
        }

        try:
            # 1. 解析 JSON（使用强化版解析器）
            parsed = _extract_json_dict(raw)
            if parsed is None:
                # 打印原始内容前 300 字符用于调试
                print(f"⚠️ JSON parse failed, raw head: {repr(str(raw)[:300])}")
                parsed = {
                    "entities": [],
                    "relations": [],
                    "relationships": [],
                    "domains": [],
                    "bridges": [],
                    "raw": str(raw)
                }

            # 2. 后处理实体（动态 Schema）
            raw_entities = parsed.get("entities", [])
            entities = post_process_entities(raw_entities, allowed_types=domain_entity_types)

            # 3. 后处理关系（动态 Schema）
            raw_relations = parsed.get("relations", [])
            # 兼容 relationships 字段
            if not raw_relations:
                raw_relations = parsed.get("relationships", [])

            relations = post_process_relations(
                raw_relations,
                entities,
                allowed_relation_types=domain_relation_types
            )

            # 4. 构建统一的 dict 结果
            result = {
                "entities": entities,
                "relations": relations,
                "relationships": relations,  # 兼容字段
                "domains": parsed.get("domains", []),
                "bridges": parsed.get("bridges", []),
                "raw": str(raw)  # 保留原始输出用于调试
            }

        except Exception as e:
            print(f"⚠️ 后处理失败，返回空结果: {e}")

        # 保存结果到缓存
        self._save_to_cache(cache_key, result)

        return result

    def _build_compatible_result(self, entities: List[Dict], relations: List[Dict]) -> str:
        """
        构建兼容旧格式的结果

        格式：
        ("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>{tuple_delimiter}<desc>){record_delimiter}
        ("relationship"{tuple_delimiter}<src>{tuple_delimiter}<tgt>{tuple_delimiter}<type>{tuple_delimiter}<desc>{tuple_delimiter}<strength>){record_delimiter}
        {completion_delimiter}
        """
        lines = []

        # 实体
        for e in entities:
            name = e.get("name", "")
            e_type = e.get("type", "")
            desc = e.get("description", name)  # 如果没有描述，使用名称
            line = f'("entity"{self.tuple_delimiter}{name}{self.tuple_delimiter}{e_type}{self.tuple_delimiter}{desc})'
            lines.append(line)

        # 关系
        for r in relations:
            src = r.get("source", "")
            tgt = r.get("target", "")
            r_type = r.get("type", "")
            desc = r.get("description", f"{src} {r_type} {tgt}")
            strength = r.get("strength", 8)  # 默认强度 8
            line = f'("relationship"{self.tuple_delimiter}{src}{self.tuple_delimiter}{tgt}{self.tuple_delimiter}{r_type}{self.tuple_delimiter}{desc}{self.tuple_delimiter}{strength})'
            lines.append(line)

        result = self.record_delimiter.join(lines) + self.completion_delimiter
        return result

    # ========== 以下方法保持不变（并行处理和缓存逻辑）==========

    def process_chunks(self, file_contents: List[Tuple], progress_callback=None) -> List[Tuple]:
        """
        并行处理所有文件的所有chunks（+ Schema-aware routing）

        新增：文件级 Domain Routing（在进入 chunk 循环前）
        """
        t0 = time.time()
        chunk_index = 0
        total_chunks = sum(len(file_content[2]) for file_content in file_contents)

        # 🔥 Step 3: 文件级 Domain Routing（新增，但不侵入）
        file_domain_schema = {}
        for file_content in file_contents:
            filename = file_content[0]  # file_content: (filename, content, chunks, ...)
            content = file_content[1]   # 原始文本内容

            # 路由领域
            domain = self._route_domain(filename, content)

            # 获取该领域的 schema
            entity_types, relation_types = self._get_schema(domain)

            # 保存到映射表
            file_domain_schema[filename] = (entity_types, relation_types)

            print(f"📋 文件 '{filename}' → Domain: {domain} (实体类型: {len(entity_types)}, 关系类型: {len(relation_types)})")

        for i, file_content in enumerate(file_contents):
            filename = file_content[0]
            chunks = file_content[2]

            # 🔥 获取该文件的 domain schema
            domain_entity_types, domain_relation_types = file_domain_schema[filename]

            # 预检查缓存命中率
            cache_keys = [self._generate_cache_key(''.join(chunk)) for chunk in chunks]
            cached_results = {key: self._load_from_cache(key) for key in cache_keys}
            non_cached_indices = [idx for idx, key in enumerate(cache_keys) if cached_results[key] is None]

            if len(non_cached_indices) > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 🔥 传递 domain schema 给 _process_single_chunk
                    future_to_chunk = {
                        executor.submit(
                            self._process_single_chunk,
                            ''.join(chunks[idx]),
                            domain_entity_types,
                            domain_relation_types
                        ): idx
                        for idx in non_cached_indices
                    }

                    for future in concurrent.futures.as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            result = future.result()
                            cached_results[cache_keys[chunk_idx]] = result

                            if progress_callback:
                                progress_callback(chunk_index)
                            chunk_index += 1

                        except Exception as exc:
                            print(f'Chunk {chunk_idx} 处理异常: {exc}')
                            retry_count = 0
                            while retry_count < 3:
                                try:
                                    print(f'尝试重试 Chunk {chunk_idx}, 第 {retry_count+1} 次')
                                    result = self._process_single_chunk(
                                        ''.join(chunks[chunk_idx]),
                                        domain_entity_types,
                                        domain_relation_types
                                    )
                                    cached_results[cache_keys[chunk_idx]] = result
                                    break
                                except Exception as retry_exc:
                                    print(f'重试失败: {retry_exc}')
                                    retry_count += 1
                                    time.sleep(1)

                            if cached_results[cache_keys[chunk_idx]] is None:
                                # ✅ 重试失败后返回空的 dict 结构，而不是字符串
                                cached_results[cache_keys[chunk_idx]] = {
                                    "entities": [],
                                    "relations": [],
                                    "relationships": [],
                                    "domains": [],
                                    "bridges": [],
                                    "raw": ""
                                }

            ordered_results = [cached_results[key] for key in cache_keys]
            file_content.append(ordered_results)

            cache_ratio = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
            print(f"文件 {i+1}/{len(file_contents)} 处理完成, 缓存命中率: {cache_ratio:.1f}%")

        process_time = time.time() - t0
        print(f"所有chunks处理完成, 总耗时: {process_time:.2f}秒, 平均每chunk: {process_time/total_chunks:.2f}秒")
        return file_contents

    def _extract_one_chunk(self, chunk_text: str, allowed_entity_types: set, allowed_relation_types: set) -> Dict[str, Any]:
        """
        包装单 chunk 抽取逻辑，供 batch 调用。

        Args:
            chunk_text: chunk 文本
            allowed_entity_types: 允许的实体类型集合
            allowed_relation_types: 允许的关系类型集合

        Returns:
            Dict: 包含 entities, relations, relationships 等字段的字典
        """
        return self._process_single_chunk(chunk_text, allowed_entity_types, allowed_relation_types)

    def process_chunks_batch(self, file_contents: List[Tuple], progress_callback=None) -> List[Tuple]:
        """
        Batch 版：保证 chunk 与 LLM 抽取结果严格对齐。

        Args:
            file_contents: [ [filename, content, chunks], ... ]

        Returns:
            [(fname, orig_chunks, proc_chunks), ...]
              - orig_chunks: List[str]  # 每个 chunk 的原文
              - proc_chunks: List[Dict]  # 每个 chunk 的抽取结果
        """

        def _pick_chunks_from_fc(fc):
            """
            从 fc 中提取 filename 和 chunk 文本列表。
            fc 可能是：
              - dict: {"filename": ..., "chunks": [...]}
              - tuple/list: [fname, content, chunks] 或其他变体
            """
            if isinstance(fc, dict):
                fname = fc.get("filename") or fc.get("fname") or fc.get("file") or ""
                chunks = fc.get("chunks") or fc.get("orig_chunks") or []
                return fname, chunks

            fname = fc[0] if len(fc) > 0 else ""

            # ✅ 优先找"像 chunk 列表"的那个字段：list 且元素是 str/list 且每个元素长度明显>1
            for idx in range(1, len(fc)):
                v = fc[idx]
                if isinstance(v, list) and v:
                    # chunk 可能是 str 或 list（带分隔符的）
                    if all(isinstance(x, str) for x in v):
                        avg_len = sum(len(x) for x in v) / max(1, len(v))
                        if avg_len >= 20:   # chunk 一般不会是单字
                            return fname, v
                    elif all(isinstance(x, list) for x in v):
                        # chunks 是 [[text, sep1, sep2], ...] 这种格式
                        # 提取每个 chunk 的文本部分
                        chunk_texts = []
                        for chunk_item in v:
                            if isinstance(chunk_item, list) and len(chunk_item) > 0:
                                # 取第一个元素作为文本，或者拼接所有元素
                                text = ''.join(chunk_item) if all(isinstance(x, str) for x in chunk_item) else str(chunk_item[0])
                                chunk_texts.append(text)
                            else:
                                chunk_texts.append(str(chunk_item))
                        if chunk_texts:
                            return fname, chunk_texts

            # 如果没找到，返回空
            return fname, []

        # 1) 提取每个文件的 chunks
        normalized = []
        for fc in file_contents:
            fname, chunks = _pick_chunks_from_fc(fc)
            normalized.append((fname, chunks))

        bad = [fn for fn, ch in normalized if not ch]
        if bad:
            raise ValueError(f"process_chunks_batch: 未能从输入中解析出 chunk 列表，出问题的文件: {bad[:5]} (共{len(bad)}个)")

        # 2) 准备每个文件的 schema（使用 GraphConfig 的 route_domain）
        file_schema_map: Dict[str, Tuple[set, set]] = {}
        graph_config = self._get_graph_config()

        for fc in file_contents:
            filename = fc[0]
            content = fc[1] if len(fc) > 1 else ""

            if graph_config:
                domain_name = graph_config.route_domain(filename, content or "")
                domain_def = graph_config.get_domain(domain_name)
                if domain_def:
                    ent_types = set(domain_def.entity_types)
                    rel_types = set(domain_def.relation_types)
                else:
                    ent_types = set(self.entity_types) or DEFAULT_ALLOWED_ENTITY_TYPES
                    rel_types = set(self.relationship_types) or DEFAULT_ALLOWED_RELATION_TYPES
            else:
                ent_types = set(self.entity_types) or DEFAULT_ALLOWED_ENTITY_TYPES
                rel_types = set(self.relationship_types) or DEFAULT_ALLOWED_RELATION_TYPES

            file_schema_map[filename] = (ent_types, rel_types)

        # 3) 扁平化所有 chunks，记录每个文件的范围
        flat_chunks = []
        flat_schemas = []
        spans = []   # (fname, start, end)
        cursor = 0

        for fname, chunks in normalized:
            start = cursor
            allowed_entity_types, allowed_relation_types = file_schema_map.get(
                fname,
                (set(self.entity_types) or DEFAULT_ALLOWED_ENTITY_TYPES,
                 set(self.relationship_types) or DEFAULT_ALLOWED_RELATION_TYPES)
            )

            for chunk_text in chunks:
                flat_chunks.append(chunk_text)
                flat_schemas.append((allowed_entity_types, allowed_relation_types))

            cursor += len(chunks)
            spans.append((fname, start, cursor))

        # 4) 批量调用 LLM 抽取（使用并发）
        print(f"开始批量抽取 {len(flat_chunks)} 个 chunks...")
        llm_results = [None] * len(flat_chunks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._extract_one_chunk,
                    flat_chunks[idx],
                    flat_schemas[idx][0],
                    flat_schemas[idx][1]
                ): idx
                for idx in range(len(flat_chunks))
            }

            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                    llm_results[idx] = res
                except Exception as e:
                    print(f'Chunk {idx} 处理异常: {e}')
                    llm_results[idx] = {
                        "entities": [],
                        "relations": [],
                        "relationships": [],
                        "domains": [],
                        "bridges": [],
                        "raw": ""
                    }

                completed += 1
                if progress_callback:
                    progress_callback(completed)

        if None in llm_results:
            raise ValueError(f"process_chunks_batch: 部分 chunk 抽取失败")

        print(f"批量抽取完成，共 {len(llm_results)} 个结果")

        # 5) 切片回填到每个文件，保证对齐
        processed_file_contents = []
        for fname, start, end in spans:
            orig_chunks = flat_chunks[start:end]
            proc_chunks = llm_results[start:end]
            processed_file_contents.append((fname, orig_chunks, proc_chunks))

        return processed_file_contents
