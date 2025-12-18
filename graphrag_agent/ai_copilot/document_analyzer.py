"""
AI Copilot - 文档分析器
分析用户文档，提取关键概念、模式和领域特征
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import Counter
import json

from langchain.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


class DocumentAnalyzer:
    """文档分析器，用于分析文档集合并提取关键信息"""

    def __init__(self, llm, embeddings):
        """
        初始化文档分析器

        Args:
            llm: 大语言模型
            embeddings: 嵌入模型
        """
        self.llm = llm
        self.embeddings = embeddings

    def analyze_documents(self, documents: List[Dict[str, Any]],
                         num_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        分析文档集合

        Args:
            documents: 文档列表，每个文档包含 filename 和 content
            num_clusters: 聚类数量，如果为None则自动确定

        Returns:
            分析结果字典
        """
        if not documents:
            return {
                "total_documents": 0,
                "clusters": [],
                "key_concepts": [],
                "statistics": {}
            }

        # 提取文本内容
        texts = [doc.get("content", "") for doc in documents]
        filenames = [doc.get("filename", f"doc_{i}") for i, doc in enumerate(documents)]

        # 1. 文档聚类
        clusters = self._cluster_documents(texts, filenames, num_clusters)

        # 2. 提取关键概念
        key_concepts = self._extract_key_concepts(texts)

        # 3. 统计信息
        statistics = {
            "total_documents": len(documents),
            "total_characters": sum(len(text) for text in texts),
            "avg_doc_length": int(np.mean([len(text) for text in texts])),
            "num_clusters": len(clusters)
        }

        return {
            "total_documents": len(documents),
            "clusters": clusters,
            "key_concepts": key_concepts,
            "statistics": statistics
        }

    def _cluster_documents(self, texts: List[str], filenames: List[str],
                          num_clusters: Optional[int] = None) -> List[Dict]:
        """
        使用 TF-IDF 和 K-Means 对文档进行聚类

        Args:
            texts: 文档文本列表
            filenames: 文档文件名列表
            num_clusters: 聚类数量

        Returns:
            聚类结果列表
        """
        if len(texts) < 2:
            return [{
                "cluster_id": 0,
                "document_count": len(texts),
                "documents": filenames,
                "representative_terms": [],
                "summary": "单文档集合"
            }]

        # 自动确定聚类数量
        if num_clusters is None:
            # 使用启发式规则：sqrt(n/2)
            num_clusters = max(2, min(5, int(np.sqrt(len(texts) / 2))))

        num_clusters = min(num_clusters, len(texts))

        try:
            # TF-IDF 向量化
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=None,  # 中文不使用英文停用词
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(texts)

            # K-Means 聚类
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            # 整理聚类结果
            clusters = []
            feature_names = vectorizer.get_feature_names_out()

            for cluster_id in range(num_clusters):
                # 获取该聚类的文档
                cluster_doc_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_filenames = [filenames[i] for i in cluster_doc_indices]

                # 获取该聚类的代表性词汇
                cluster_center = kmeans.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-10:][::-1]
                representative_terms = [feature_names[i] for i in top_indices]

                clusters.append({
                    "cluster_id": cluster_id,
                    "document_count": len(cluster_filenames),
                    "documents": cluster_filenames,
                    "representative_terms": representative_terms,
                    "summary": f"聚类 {cluster_id + 1}"
                })

            return clusters

        except Exception as e:
            print(f"聚类失败: {e}")
            # 回退：所有文档归为一个聚类
            return [{
                "cluster_id": 0,
                "document_count": len(texts),
                "documents": filenames,
                "representative_terms": [],
                "summary": "未聚类集合"
            }]

    def _extract_key_concepts(self, texts: List[str], top_n: int = 20) -> List[Dict]:
        """
        提取文档集合的关键概念

        Args:
            texts: 文档文本列表
            top_n: 返回前N个关键概念

        Returns:
            关键概念列表
        """
        try:
            # 使用 TF-IDF 提取关键词
            vectorizer = TfidfVectorizer(
                max_features=top_n,
                stop_words=None,
                ngram_range=(1, 3)  # 支持1-3个词的短语
            )
            tfidf_matrix = vectorizer.fit_transform(texts)

            # 计算每个词的平均 TF-IDF 分数
            feature_names = vectorizer.get_feature_names_out()
            avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)

            # 排序并返回
            concept_scores = list(zip(feature_names, avg_tfidf))
            concept_scores.sort(key=lambda x: x[1], reverse=True)

            return [
                {
                    "concept": concept,
                    "score": float(score),
                    "type": "keyword"  # 后续可以扩展为实体类型
                }
                for concept, score in concept_scores[:top_n]
            ]

        except Exception as e:
            print(f"关键概念提取失败: {e}")
            return []

    def recommend_domains_and_bridges(self, analysis_result: Dict[str, Any],
                                     industry_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        基于文档分析结果，使用 LLM 推荐领域和桥接点

        Args:
            analysis_result: 文档分析结果
            industry_hint: 行业提示（可选）

        Returns:
            推荐结果
        """
        # 构建提示词
        prompt = self._build_recommendation_prompt(analysis_result, industry_hint)

        # 调用 LLM
        try:
            response = self.llm.invoke(prompt)
            content = response.content

            # 尝试解析 JSON
            # 移除可能的代码块标记
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}, 内容: {content}")
            # 返回默认结构
            return {
                "recommended_bridges": [],
                "recommended_domains": [],
                "reasoning": "LLM 响应解析失败"
            }
        except Exception as e:
            print(f"推荐生成失败: {e}")
            return {
                "recommended_bridges": [],
                "recommended_domains": [],
                "reasoning": f"推荐失败: {str(e)}"
            }

    def _build_recommendation_prompt(self, analysis_result: Dict[str, Any],
                                    industry_hint: Optional[str] = None) -> str:
        """构建推荐提示词"""
        clusters = analysis_result.get("clusters", [])
        key_concepts = analysis_result.get("key_concepts", [])
        statistics = analysis_result.get("statistics", {})

        prompt = f"""# 任务：基于文档分析推荐知识图谱配置

## 文档分析结果

**统计信息**：
- 文档数量：{statistics.get('total_documents', 0)}
- 总字符数：{statistics.get('total_characters', 0)}
- 平均文档长度：{statistics.get('avg_doc_length', 0)}
- 聚类数量：{statistics.get('num_clusters', 0)}

**文档聚类**：
"""

        for cluster in clusters[:5]:  # 最多显示5个聚类
            prompt += f"\n聚类 {cluster['cluster_id'] + 1}（{cluster['document_count']} 个文档）:\n"
            prompt += f"- 代表性词汇：{', '.join(cluster['representative_terms'][:10])}\n"
            prompt += f"- 包含文档：{', '.join(cluster['documents'][:5])}\n"

        prompt += f"\n**关键概念**：\n"
        for concept in key_concepts[:15]:
            prompt += f"- {concept['concept']} (权重: {concept['score']:.3f})\n"

        if industry_hint:
            prompt += f"\n**行业提示**：{industry_hint}\n"

        prompt += """

## 你的任务

基于以上分析，推荐：
1. **桥接点（Bridges）**：跨领域的公共概念，用于连接不同类型的文档
   - 例如：问题类型、风险等级、状态、优先级等
   - 每个桥接点应该有明确的业务含义
   - 建议 2-4 个桥接点

2. **领域（Domains）**：业务子图，根据文档聚类结果定义
   - 每个聚类可能对应一个领域
   - 每个领域应该有明确的业务场景
   - 为每个领域推荐实体类型和关系类型

## 输出格式

请严格按照以下 JSON 格式输出（不要添加任何其他文字）：

```json
{
  "recommended_bridges": [
    {
      "name": "桥接点名称",
      "key": "bridge_xxx",
      "description": "桥接点描述",
      "examples": ["示例1", "示例2"],
      "is_enum_restricted": false,
      "reasoning": "为什么推荐这个桥接点"
    }
  ],
  "recommended_domains": [
    {
      "domain_name": "领域名称",
      "description": "领域描述",
      "trigger_condition": "触发条件描述",
      "schema": {
        "entities": ["实体类型1", "实体类型2"],
        "relations": ["关系类型1", "关系类型2"]
      },
      "bridge_mappings": [
        {
          "bridge_key": "bridge_xxx",
          "role": "角色描述"
        }
      ],
      "reasoning": "为什么推荐这个领域"
    }
  ],
  "reasoning": "整体推荐理由和建议"
}
```

现在请开始分析并输出推荐结果：
"""

        return prompt

    def refine_recommendations(self, current_config: Dict[str, Any],
                             user_feedback: str) -> Dict[str, Any]:
        """
        基于用户反馈优化推荐

        Args:
            current_config: 当前配置
            user_feedback: 用户反馈

        Returns:
            优化后的推荐
        """
        prompt = f"""# 任务：优化知识图谱配置

## 当前配置

```json
{json.dumps(current_config, ensure_ascii=False, indent=2)}
```

## 用户反馈

{user_feedback}

## 你的任务

根据用户反馈，调整和优化当前配置。请输出完整的优化后配置。

输出格式（JSON）：

```json
{{
  "bridge_definitions": [...],
  "domain_definitions": [...],
  "changes_summary": "主要修改说明"
}}
```

现在请输出优化结果：
"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # 移除代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            return result

        except Exception as e:
            print(f"优化失败: {e}")
            return {
                "bridge_definitions": current_config.get("bridge_definitions", []),
                "domain_definitions": current_config.get("domain_definitions", []),
                "changes_summary": f"优化失败: {str(e)}"
            }
