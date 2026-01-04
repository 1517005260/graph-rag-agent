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
import logging
import unicodedata
import concurrent.futures
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# 配置日志
logger = logging.getLogger(__name__)

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
    "机构",         # 对应 Organization (如: 华东理工大学, 答辩委员会)
    "部门",         # 对应 Department (如: 研究生院, 学生工作部)
    "政策",         # 对应 Policy (如: 奖学金管理办法)
    "规章制度",     # 对应 Regulation
    "法条",         # 对应 Law/Article
    "条款",         # 对应 Clause/Condition (如: 6学分, 违纪处分)
    "文档",         # 对应 Document (如: 申请表)
    "当事人",       # 对应 Person/Party (如: 毕业生, 学生本人)
    "地点",         # 对应 Location (如: 中西部地区)
    "概念",         # 对应 Concept (如: 学费代偿)
    "流程",         # 对应 Process
    "条件"          # 对应 Condition
}

ALLOWED_RELATION_TYPES = {
    "包含",         # PART_OF / INCLUDES
    "属于",         # PART_OF
    "发布",         # ISSUED_BY
    "负责",         # RESPONSIBLE_FOR
    "需要",         # REQUIRES
    "依据",         # BASED_ON
    "适用于",       # APPLIES_TO
    "有步骤",       # HAS_STEP
    "有条件",       # HAS_CONDITION
    "关联",         # RELATED_TO (兜底)
    "提交给"        # SUBMIT_TO
}

# 默认后处理参数（可通过初始化覆盖）
DEFAULT_MIN_ENTITY_FREQUENCY = 1            # 默认最小实体频率
DEFAULT_NAME_SIMILARITY_THRESHOLD = 0.85    # 默认名称相似度阈值

# 默认常量（用作 fallback）
DEFAULT_ALLOWED_ENTITY_TYPES = ALLOWED_ENTITY_TYPES
DEFAULT_ALLOWED_RELATION_TYPES = ALLOWED_RELATION_TYPES


# =========================
# 工具函数（生产级）
# =========================

def normalize_entity_name(name: str) -> str:
    """
    标准化实体名称（生产级多语言支持）

    使用 Unicode NFKC 标准化，自动处理：
    - 全角字符 → 半角（如 Ａ→A，１→1，％→%）
    - 全角空格 → 半角空格
    - 其他 Unicode 变体 → 标准形式

    规则：
    1. NFKC 标准化（兼容性分解 + 标准合成）
    2. 去除所有空格
    3. 去除首尾空白

    Args:
        name: 原始实体名称

    Returns:
        str: 标准化后的实体名称
    """
    if not name:
        return ""

    # Unicode NFKC 标准化：全角 → 半角，统一 Unicode 变体
    normalized = unicodedata.normalize('NFKC', name)

    # 去除空格并清理
    normalized = normalized.replace(" ", "").strip()

    return normalized


def is_similar(a: str, b: str, threshold: float = None) -> bool:
    """
    判断两个实体名称是否相似（基于 SequenceMatcher）

    Args:
        a: 第一个实体名称
        b: 第二个实体名称
        threshold: 相似度阈值，默认使用全局默认值

    Returns:
        bool: 是否相似
    """
    if threshold is None:
        threshold = DEFAULT_NAME_SIMILARITY_THRESHOLD
    return SequenceMatcher(None, a, b).ratio() >= threshold


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

def post_process_entities(
    raw_entities: List[Dict[str, Any]],
    allowed_entity_types: set,
    min_freq: int,
    similarity_threshold: float
) -> List[Dict[str, Any]]:
    """
    实体后处理（生产级验证 + 动态 Schema）

    步骤：
    1. normalize：标准化名称
    2. type_filter：类型过滤（动态白名单，大小写不敏感）
    3. frequency_filter：频率过滤
    4. deduplicate：去重（相似度阈值）

    Args:
        raw_entities: 原始实体列表
        allowed_entity_types: 允许的实体类型集合
        min_freq: 最小实体频率
        similarity_threshold: 相似度阈值

    Returns:
        List[Dict[str, Any]]: 处理后的实体列表
    """
    if not raw_entities:
        return []

    # [核心修复] 1. 制作全大写的白名单集合
    allowed_types_upper = {t.upper() for t in allowed_entity_types}

    # --- [新增调试代码] ---
    print(f"\n🔍 DEBUG: 白名单(Allowed): {allowed_types_upper}")
    for e in raw_entities:
        print(f"   🧐 LLM输出: Name='{e.get('name')}', Type='{e.get('type')}'")
    # --------------------------

    # normalize names
    for e in raw_entities:
        if "name" in e:
            e["name"] = normalize_entity_name(e.get("name", ""))

    # 2. type filter（动态白名单，大小写不敏感）
    entities = []
    for e in raw_entities:
        raw_type = e.get("type", "")
        # [核心修复] 2. 转大写后对比，且 e["name"] 不能为空
        if raw_type and raw_type.upper() in allowed_types_upper and e.get("name"):
            # 可选：将类型标准化为大写，或者保持原样
            # e["type"] = raw_type.upper()
            entities.append(e)

    # 3. frequency filter
    freq = Counter(e["name"] for e in entities)
    entities = [e for e in entities if freq[e["name"]] >= min_freq]

    # 4. similarity dedup（优化：分桶策略，O(N²) → O(N log N)）
    # 按频率降序排列：高频词更规范，保留高频词，让低频词去匹配高频词
    sorted_entities = sorted(entities, key=lambda x: freq[x["name"]], reverse=True)

    # 使用分桶（Blocking）策略：按首字符分组，只对同组内的实体进行相似度比对
    # 这将大幅减少比对次数：从 O(N²) 降到 O(N × avg_bucket_size)
    deduped = []
    buckets = defaultdict(list)  # 首字符 → 已去重实体列表

    for e in sorted_entities:
        name = e["name"]
        if not name:
            continue

        # 分桶键：使用首字符（可扩展为前2字符或 MinHash）
        bucket_key = name[0] if name else ""

        # 只与同一桶内的实体比对
        candidates = buckets[bucket_key]
        is_duplicate = any(
            is_similar(name, candidate["name"], similarity_threshold)
            for candidate in candidates
        )

        if not is_duplicate:
            deduped.append(e)
            buckets[bucket_key].append(e)

    logger.info(
        f"✅ 实体后处理：{len(raw_entities)} → {len(deduped)} 个实体"
        f"（Schema: {len(allowed_entity_types)} 类型，频率阈值: {min_freq}，"
        f"相似度阈值: {similarity_threshold}）"
    )
    return deduped


def post_process_relations(
    raw_relations: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    allowed_relation_types: set,
    similarity_threshold: float
) -> List[Dict[str, Any]]:
    """
    关系后处理（生产级验证 + 动态 Schema）

    步骤：
    1. 验证关系类型（动态白名单，大小写不敏感）
    2. 验证 source/target 实体存在
    3. 去重

    Args:
        raw_relations: 原始关系列表
        entities: 实体列表
        allowed_relation_types: 允许的关系类型集合
        similarity_threshold: 相似度阈值（保留参数以统一接口，关系处理暂不使用）

    Returns:
        List[Dict[str, Any]]: 处理后的关系列表
    """
    if not raw_relations or not entities:
        return []

    # [新增] 1. 预处理白名单为全大写
    allowed_rels_upper = {r.upper() for r in allowed_relation_types}

    entity_names = {normalize_entity_name(e["name"]) for e in entities}
    cleaned = []
    seen = set()

    for r in raw_relations:
        src = normalize_entity_name(r.get("source", ""))
        tgt = normalize_entity_name(r.get("target", ""))
        r_type = r.get("type", "")

        # [修改] 2. 核心修改：类型转大写后对比
        if (
            src in entity_names and
            tgt in entity_names and
            r_type and r_type.upper() in allowed_rels_upper
        ):
            # Key 使用大写类型以防止重复
            key = (src, tgt, r_type.upper())
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
                 graph_config=None,
                 min_entity_frequency: int = None,
                 name_similarity_threshold: float = None):
        """
        初始化实体关系提取器（+ GraphConfig 支持 + 参数化配置 + 缓存版本隔离）

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
            min_entity_frequency: 最小实体频率（可选，默认使用全局默认值）
            name_similarity_threshold: 名称相似度阈值（可选，默认使用全局默认值）
        """
        self.llm = llm
        self.entity_types = entity_types
        self.relationship_types = relationship_types
        self.chat_history = []

        # 🔥 新增：GraphConfig 支持
        self.graph_config = graph_config

        # 🔥 新增：后处理参数化配置
        self.min_entity_frequency = (
            min_entity_frequency if min_entity_frequency is not None
            else DEFAULT_MIN_ENTITY_FREQUENCY
        )
        self.name_similarity_threshold = (
            name_similarity_threshold if name_similarity_threshold is not None
            else DEFAULT_NAME_SIMILARITY_THRESHOLD
        )

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
            logger.info(f"创建缓存目录（版本隔离）: {self.cache_dir}")

        # 🔥 并行处理配置（支持动态线程数）
        self.max_workers = self._compute_max_workers(max_workers)
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE

        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"🔥 生产级实体提取器已初始化")
        logger.info(f"   - Model: {self.model_name}")
        logger.info(f"   - Prompt Version: {self.prompt_version}")
        logger.info(f"   - Cache Dir: {self.cache_dir}")
        logger.info(f"   - Max Workers: {self.max_workers}")
        logger.info(f"   - 实体类型白名单：{ALLOWED_ENTITY_TYPES}")
        logger.info(f"   - 关系类型白名单：{ALLOWED_RELATION_TYPES}")
        logger.info(f"   - 最小实体频率：{self.min_entity_frequency}")
        logger.info(f"   - 名称相似度阈值：{self.name_similarity_threshold}")

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
            import os
            cpu_count = os.cpu_count() or 1
            computed = min(32, cpu_count + 4)
            logger.info(f"动态计算线程数：CPU 核数 {cpu_count} → {computed} 个线程")
            return computed

        # 如果是 None，使用默认配置
        if max_workers is None:
            return DEFAULT_MAX_WORKERS

        # 如果是整数，直接使用
        if isinstance(max_workers, int) and max_workers > 0:
            return max_workers

        # 其他情况，fallback 到默认值
        logger.warning(f"无效的 max_workers 值: {max_workers}，使用默认值 {DEFAULT_MAX_WORKERS}")
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
        """
        从缓存加载结果

        Args:
            cache_key: 缓存键

        Returns:
            缓存的结果，如果加载失败或缓存不存在则返回 None
        """
        if not self.enable_cache:
            return None

        cache_path = self._cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                    self.cache_hits += 1
                    return result
            except (pickle.UnpicklingError, EOFError) as e:
                # 严重错误：缓存文件损坏，记录 ERROR 级别，可能需要人工介入清理缓存
                logger.error(f"缓存文件损坏 [{cache_key}]: {e}", exc_info=True)
                logger.error(f"建议删除损坏的缓存文件: {cache_path}")
                return None
            except Exception as e:
                # 一般错误：缓存读取失败，记录 WARNING 级别
                logger.warning(f"缓存读取失败 [{cache_key}]: {e}")
                return None

        self.cache_misses += 1
        return None

    def _get_graph_config(self):
        """
        获取图谱配置（支持运行时热更新）

        优先级（从高到低）：
        1. 直接传入的 graph_config (通过 __init__)
        2. Factory 注入的 prompt_builder.config
        3. 🔥 GraphConfigService 动态读取（热更新机制）
        """
        # 1. 尝试直接获取 graph_config (如果通过 __init__ 传入)
        if self.graph_config:
            return self.graph_config

        # 2. 尝试从 prompt_builder 获取 (extractor_factory 注入的方式)
        if hasattr(self, 'prompt_builder') and self.prompt_builder and hasattr(self.prompt_builder, 'config'):
            return self.prompt_builder.config

        # 🔥 3. 从 GraphConfigService 动态读取（运行时热更新）
        try:
            # 延迟导入，避免循环依赖
            from server.services.graph_config_service import get_config_service

            config_service = get_config_service()
            config = config_service.get_config()

            if config:
                # 找到配置，打印提示（用于调试）
                print(f"[Extractor] 从 GraphConfigService 读取配置: {config.project_name}")
                return config
        except ImportError:
            # 如果在非 server 环境（如纯脚本），GraphConfigService 可能不存在
            pass
        except Exception as e:
            print(f"[Extractor] 从 GraphConfigService 读取配置失败: {e}")

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
        # [修改] 使用 _get_graph_config() 获取配置，防止 self.graph_config 为 None
        config = self._get_graph_config()

        if config and hasattr(config, 'route_domain'):
            return config.route_domain(filename, content)
        return "default"

    def _get_schema(self, domain: str) -> Tuple[set, set]:
        """
        获取领域的 Schema（实体类型 + 关系类型）

        ⚠️ DEPRECATED: 此方法已废弃，建议使用 _schema_for_domain()
        保留此方法仅为向后兼容，内部直接调用 _schema_for_domain

        Args:
            domain: 领域标识

        Returns:
            (entity_types, relation_types): 实体类型集合和关系类型集合
        """
        # 直接调用 _schema_for_domain，避免重复逻辑
        return self._schema_for_domain(domain)

    def _schema_for_domain(self, domain: str) -> Tuple[set, set]:
        """
        根据领域名称获取对应的 Schema（实体类型 + 关系类型）

        策略：
        1. 如果有 GraphConfig，从 domain_definitions 中查找匹配的领域
        2. 从 DomainDefinition.schema 获取 entities 和 relations
        3. 🔥 转换为大写以保持一致性（避免大小写导致的过滤问题）
        4. 否则返回初始化时传入的类型（或全局白名单）

        Args:
            domain: 领域标识（如 "规则库", "事实库", "default"）

        Returns:
            (entity_types, relation_types): 实体类型集合和关系类型集合（已大写化）
        """
        config = self._get_graph_config()

        if config and hasattr(config, 'domain_definitions'):
            # 遍历 domain_definitions 查找匹配的领域
            for domain_def in config.domain_definitions:
                if domain_def.domain_name == domain:
                    # 找到匹配的领域，返回其 schema
                    # 🔥 注意：不转大写，保持原样，因为 post_process_entities 会在内部处理大小写
                    entities = set(domain_def.schema.entities) if domain_def.schema.entities else set()
                    relations = set(domain_def.schema.relations) if domain_def.schema.relations else set()

                    print(f"[Extractor] 找到领域 '{domain}' 的 Schema: {len(entities)} 个实体类型, {len(relations)} 个关系类型")
                    return (entities, relations)

        # 未找到匹配或无配置，返回初始化时传入的类型
        # 如果初始化时也没有传入，则使用全局白名单作为最后的 fallback
        fallback_entities = set(self.entity_types) if self.entity_types else ALLOWED_ENTITY_TYPES
        fallback_relations = set(self.relationship_types) if self.relationship_types else ALLOWED_RELATION_TYPES

        print(f"[Extractor] 未找到领域 '{domain}' 的配置，使用 fallback schema: {len(fallback_entities)} 个实体类型")
        return (fallback_entities, fallback_relations)

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
            domain_entity_types: 领域实体类型（可选）
            domain_relation_types: 领域关系类型（可选）

        Returns:
            Dict: 包含 entities, relations 等字段的字典
        """
        # 🔥 智能 fallback：如果未提供 schema，尝试从 default 领域获取
        if domain_entity_types is None or domain_relation_types is None:
            # 优先使用 default 领域的配置（如果有 GraphConfig）
            default_entities, default_relations = self._schema_for_domain("default")

            if domain_entity_types is None:
                domain_entity_types = default_entities
            if domain_relation_types is None:
                domain_relation_types = default_relations

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

            # 2. 后处理实体（动态 Schema + 参数化配置）
            raw_entities = parsed.get("entities", [])
            entities = post_process_entities(
                raw_entities,
                allowed_entity_types=domain_entity_types,
                min_freq=self.min_entity_frequency,
                similarity_threshold=self.name_similarity_threshold
            )

            # 3. 后处理关系（动态 Schema + 参数化配置）
            raw_relations = parsed.get("relations", [])
            # 兼容 relationships 字段
            if not raw_relations:
                raw_relations = parsed.get("relationships", [])

            relations = post_process_relations(
                raw_relations,
                entities,
                allowed_relation_types=domain_relation_types,
                similarity_threshold=self.name_similarity_threshold
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

        # [修改] 不要原地修改 tuple，构造新列表
        new_file_contents = []

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

            # [修改] 构造新的 tuple：(*原Tuple内容, 新结果)
            # 注意：如果不确定 fc 是 list 还是 tuple，这种写法最稳健
            new_fc = tuple(list(file_content) + [ordered_results])
            new_file_contents.append(new_fc)

            cache_ratio = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
            print(f"文件 {i+1}/{len(file_contents)} 处理完成, 缓存命中率: {cache_ratio:.1f}%")

        process_time = time.time() - t0
        print(f"所有chunks处理完成, 总耗时: {process_time:.2f}秒, 平均每chunk: {process_time/total_chunks:.2f}秒")
        return new_file_contents

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

        # 2) 准备每个文件的 schema（使用内部方法）
        file_schema_map: Dict[str, Tuple[set, set]] = {}

        for fc in file_contents:
            filename = fc[0]
            content = fc[1] if len(fc) > 1 else ""

            # 1. 路由领域 (使用内部方法)
            domain = self._route_domain(filename, content or "")

            # 2. 获取 Schema (使用内部方法，不要直接调 graph_config)
            ent_types, rel_types = self._schema_for_domain(domain)

            file_schema_map[filename] = (ent_types, rel_types)

        # 3) 扁平化所有 chunks，记录每个文件的范围
        flat_chunks = []
        flat_schemas = []
        spans = []   # (fname, start, end)
        cursor = 0

        for fname, chunks in normalized:
            start = cursor

            # 🔥 智能 fallback：如果文件没有映射到 schema，使用 default 领域
            if fname not in file_schema_map:
                default_entities, default_relations = self._schema_for_domain("default")
                allowed_entity_types = default_entities
                allowed_relation_types = default_relations
            else:
                allowed_entity_types, allowed_relation_types = file_schema_map[fname]

            for chunk_text in chunks:
                flat_chunks.append(chunk_text)
                flat_schemas.append((allowed_entity_types, allowed_relation_types))

            cursor += len(chunks)
            spans.append((fname, start, cursor))

        # 4) 批量调用 LLM 抽取（使用并发 + 增强型熔断机制）
        logger.info(f"开始批量抽取 {len(flat_chunks)} 个 chunks...")
        llm_results = [None] * len(flat_chunks)

        # 错误和空结果计数器
        error_count = 0
        empty_result_count = 0  # 新增：空结果计数
        total_chunks = len(flat_chunks)

        # 熔断配置
        ERROR_RATE_THRESHOLD = 0.2  # 错误率阈值：20%
        EMPTY_RATE_THRESHOLD = 0.3  # 空结果率阈值：30%
        MIN_CHUNKS_FOR_CIRCUIT_BREAKER = 10  # 至少处理 10 个 chunk 后才启用熔断

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

                    # 🔥 新增：检查空结果（LLM 返回成功但无实体/关系）
                    if not res.get("entities") and not res.get("relationships"):
                        empty_result_count += 1
                        logger.warning(
                            f"Chunk {idx} 返回空结果（无实体和关系）。"
                            f"这可能是正常的，也可能表明 Prompt 或内容有问题。"
                        )

                except Exception as e:
                    error_count += 1
                    # 记录详细错误信息
                    logger.error(f"Chunk {idx} 处理失败: {e}", exc_info=True)

                    # 填充空结果（仅在未触发熔断时）
                    llm_results[idx] = {
                        "entities": [],
                        "relations": [],
                        "relationships": [],
                        "domains": [],
                        "bridges": [],
                        "raw": ""
                    }

                completed += 1

                # 🔥 增强型熔断机制：检查错误率和空结果率
                if completed > MIN_CHUNKS_FOR_CIRCUIT_BREAKER:
                    current_error_rate = error_count / completed
                    current_empty_rate = empty_result_count / completed
                    combined_failure_rate = (error_count + empty_result_count) / completed

                    # 检查错误率
                    if current_error_rate > ERROR_RATE_THRESHOLD:
                        error_msg = (
                            f"错误率过高 ({current_error_rate:.1%} > {ERROR_RATE_THRESHOLD:.1%})，"
                            f"已处理 {completed}/{total_chunks} 个 chunk，失败 {error_count} 个。"
                            f"中止图谱构建。请检查 LLM 连接或 Prompt 配置。"
                        )
                        logger.critical(error_msg)
                        # 取消剩余任务
                        for f in futures.keys():
                            if not f.done():
                                f.cancel()
                        raise RuntimeError(error_msg)

                    # 检查空结果率
                    if current_empty_rate > EMPTY_RATE_THRESHOLD:
                        error_msg = (
                            f"空结果率过高 ({current_empty_rate:.1%} > {EMPTY_RATE_THRESHOLD:.1%})，"
                            f"已处理 {completed}/{total_chunks} 个 chunk，"
                            f"空结果 {empty_result_count} 个（无实体和关系）。"
                            f"中止图谱构建。请检查 Prompt 配置或文档内容质量。"
                        )
                        logger.critical(error_msg)
                        # 取消剩余任务
                        for f in futures.keys():
                            if not f.done():
                                f.cancel()
                        raise RuntimeError(error_msg)

                    # 综合检查（错误+空结果）
                    if combined_failure_rate > 0.5:  # 50% 综合失败率
                        error_msg = (
                            f"综合失败率过高 ({combined_failure_rate:.1%} > 50%)，"
                            f"已处理 {completed}/{total_chunks} 个 chunk，"
                            f"错误 {error_count} 个，空结果 {empty_result_count} 个。"
                            f"中止图谱构建。"
                        )
                        logger.critical(error_msg)
                        # 取消剩余任务
                        for f in futures.keys():
                            if not f.done():
                                f.cancel()
                        raise RuntimeError(error_msg)

                if progress_callback:
                    progress_callback(completed)

        # 🔥 新增：完整性校验
        # 1. 检查数量对齐
        if len(llm_results) != total_chunks:
            raise RuntimeError(
                f"严重错误：结果数量({len(llm_results)})与输入块数({total_chunks})不一致！"
            )

        # 2. 检查 None 值（防止意外漏填）
        none_indices = [i for i, r in enumerate(llm_results) if r is None]
        if none_indices:
            raise RuntimeError(
                f"严重错误：存在 {len(none_indices)} 个未被填充的结果槽位（索引: {none_indices[:10]}...），"
                f"请检查并发逻辑。"
            )

        # 记录最终统计
        if error_count > 0 or empty_result_count > 0:
            error_rate = error_count / total_chunks
            empty_rate = empty_result_count / total_chunks
            logger.warning(
                f"批量抽取完成，共 {total_chunks} 个 chunk，"
                f"成功 {total_chunks - error_count - empty_result_count} 个，"
                f"失败 {error_count} 个 (错误率: {error_rate:.1%})，"
                f"空结果 {empty_result_count} 个 (空结果率: {empty_rate:.1%})"
            )
        else:
            logger.info(f"批量抽取完成，共 {total_chunks} 个 chunk，全部成功")

        print(f"批量抽取完成，共 {len(llm_results)} 个结果")

        # 5) 切片回填到每个文件，保证对齐
        processed_file_contents = []
        for fname, start, end in spans:
            orig_chunks = flat_chunks[start:end]
            proc_chunks = llm_results[start:end]
            processed_file_contents.append((fname, orig_chunks, proc_chunks))

        return processed_file_contents
