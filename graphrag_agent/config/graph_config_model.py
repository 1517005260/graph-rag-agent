"""
通用图谱配置模型
支持多领域、多租户的知识图谱配置
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class BridgeDefinition(BaseModel):
    """桥接点定义 - 连接不同领域的公共概念"""
    name: str = Field(..., description="桥接点名称，如'核心问题类型'")
    key: str = Field(..., description="桥接点唯一标识，如'bridge_issue_type'")
    description: str = Field(..., description="桥接点的含义说明")
    examples: List[str] = Field(default_factory=list, description="示例值，用于引导AI提取")
    is_enum_restricted: bool = Field(default=False, description="是否限制在枚举值范围内")
    enum_values: Optional[List[str]] = Field(default=None, description="枚举值列表")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "问题类型",
                "key": "bridge_issue_type",
                "description": "连接所有文档的核心问题分类",
                "examples": ["投诉", "咨询", "建议"],
                "is_enum_restricted": True,
                "enum_values": ["投诉", "咨询", "建议", "举报"]
            }
        }


class BridgeMapping(BaseModel):
    """桥接点映射 - 定义领域如何连接到桥接点"""
    bridge_key: str = Field(..., description="桥接点的 key")
    role: str = Field(..., description="该桥接点在此领域中的角色，如'分类标签'、'严重程度'")
    field_name: Optional[str] = Field(default=None, description="该桥接点在此领域中的别名")


class DomainSchema(BaseModel):
    """领域 Schema - 定义领域内的实体和关系"""
    entities: List[str] = Field(..., description="该领域特有的实体类型")
    relations: List[str] = Field(..., description="该领域特有的关系类型")


class DomainDefinition(BaseModel):
    """领域定义 - 业务子图"""
    domain_name: str = Field(..., description="领域名称，如'规则库'、'事实库'")
    description: str = Field(..., description="领域的用途说明")
    trigger_condition: str = Field(..., description="AI 用于判断文档属于该领域的条件")
    schema: DomainSchema = Field(..., description="该领域的实体和关系定义")
    bridge_mappings: List[BridgeMapping] = Field(default_factory=list, description="桥接点映射")

    class Config:
        json_schema_extra = {
            "example": {
                "domain_name": "规则库",
                "description": "作为判断依据的标准文档，如法律法规、公司政策",
                "trigger_condition": "当文档包含条款、规定、参数表、标准时",
                "schema": {
                    "entities": ["法条", "条款", "参数", "标准"],
                    "relations": ["规定", "约束", "定义", "包含"]
                },
                "bridge_mappings": [
                    {"bridge_key": "bridge_issue_type", "role": "适用场景"}
                ]
            }
        }


class GraphConfig(BaseModel):
    """通用图谱配置 - 完整的多领域配置"""
    project_name: str = Field(..., description="项目名称")
    version: str = Field(default="1.0", description="配置版本")
    description: Optional[str] = Field(default=None, description="项目描述")
    industry: Optional[str] = Field(default=None, description="所属行业，如'法务'、'电商'、'医疗'")

    # 核心配置
    bridge_definitions: List[BridgeDefinition] = Field(..., description="桥接点定义列表")
    domain_definitions: List[DomainDefinition] = Field(..., description="领域定义列表")

    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "project_name": "XX市信访系统",
                "version": "1.0",
                "description": "用于处理市民信访案件的知识图谱",
                "industry": "法务",
                "bridge_definitions": [
                    {
                        "name": "问题类型",
                        "key": "bridge_issue_type",
                        "description": "信访问题的分类",
                        "examples": ["欠薪纠纷", "房屋质量", "环境污染"],
                        "is_enum_restricted": False
                    }
                ],
                "domain_definitions": [
                    {
                        "domain_name": "规则库",
                        "description": "法律法规和政策文件",
                        "trigger_condition": "包含法律条款、政策规定",
                        "schema": {
                            "entities": ["法条", "政策", "部门"],
                            "relations": ["规定", "负责", "依据"]
                        },
                        "bridge_mappings": [
                            {"bridge_key": "bridge_issue_type", "role": "适用场景"}
                        ]
                    }
                ]
            }
        }


# ==================== 行业模板 ====================

class IndustryTemplate:
    """行业模板库"""

    @staticmethod
    def get_legal_template() -> GraphConfig:
        """信访/法务模板"""
        return GraphConfig(
            project_name="信访法务系统",
            industry="法务",
            description="适用于信访、法律咨询、案件管理等场景",
            bridge_definitions=[
                BridgeDefinition(
                    name="问题类型",
                    key="bridge_issue_type",
                    description="案件或信访的核心问题分类",
                    examples=["劳动纠纷", "房产纠纷", "环境问题", "民事纠纷"],
                    is_enum_restricted=False
                ),
                BridgeDefinition(
                    name="风险等级",
                    key="bridge_risk_level",
                    description="案件的风险或优先级",
                    examples=["高风险", "中风险", "低风险", "一般"],
                    is_enum_restricted=True,
                    enum_values=["高风险", "中风险", "低风险", "一般"]
                )
            ],
            domain_definitions=[
                DomainDefinition(
                    domain_name="法规库",
                    description="法律法规、政策文件、规章制度",
                    trigger_condition="文档包含法律条款、政策规定、规章制度",
                    schema=DomainSchema(
                        entities=["法条", "政策", "条款", "部门", "机构"],
                        relations=["规定", "依据", "负责", "管辖", "约束"]
                    ),
                    bridge_mappings=[
                        BridgeMapping(bridge_key="bridge_issue_type", role="适用场景")
                    ]
                ),
                DomainDefinition(
                    domain_name="案件库",
                    description="历史案件、信访记录、诉讼档案",
                    trigger_condition="文档包含案件描述、当事人、时间地点等要素",
                    schema=DomainSchema(
                        entities=["当事人", "案件", "行为", "地点", "时间", "证据"],
                        relations=["投诉", "涉及", "发生于", "导致", "解决"]
                    ),
                    bridge_mappings=[
                        BridgeMapping(bridge_key="bridge_issue_type", role="案由"),
                        BridgeMapping(bridge_key="bridge_risk_level", role="紧急程度")
                    ]
                ),
                DomainDefinition(
                    domain_name="经验库",
                    description="处理经验、调解方案、判例分析",
                    trigger_condition="文档包含总结、经验、建议、方案",
                    schema=DomainSchema(
                        entities=["处理方案", "调解方法", "建议", "经验"],
                        relations=["适用于", "解决", "参考", "借鉴"]
                    ),
                    bridge_mappings=[
                        BridgeMapping(bridge_key="bridge_issue_type", role="适用问题")
                    ]
                )
            ]
        )

    @staticmethod
    def get_ecommerce_template() -> GraphConfig:
        """电商客服模板"""
        return GraphConfig(
            project_name="电商客服系统",
            industry="电商",
            description="适用于电商客服、售后服务、用户工单管理",
            bridge_definitions=[
                BridgeDefinition(
                    name="产品型号",
                    key="bridge_product_sku",
                    description="商品的唯一标识或型号",
                    examples=["iPhone 15 Pro", "小米手机13"],
                    is_enum_restricted=False
                ),
                BridgeDefinition(
                    name="问题类型",
                    key="bridge_issue_type",
                    description="客户问题或工单类型",
                    examples=["退货退款", "质量问题", "物流咨询", "使用指导"],
                    is_enum_restricted=True,
                    enum_values=["退货退款", "质量问题", "物流咨询", "使用指导", "投诉建议"]
                )
            ],
            domain_definitions=[
                DomainDefinition(
                    domain_name="产品手册",
                    description="商品说明书、参数规格、使用手册",
                    trigger_condition="文档包含产品参数、规格、使用说明",
                    schema=DomainSchema(
                        entities=["产品", "参数", "规格", "功能", "配件"],
                        relations=["具有", "支持", "包含", "兼容"]
                    ),
                    bridge_mappings=[
                        BridgeMapping(bridge_key="bridge_product_sku", role="产品标识")
                    ]
                ),
                DomainDefinition(
                    domain_name="工单记录",
                    description="客户咨询、投诉、售后工单",
                    trigger_condition="文档包含客户对话、问题描述、处理记录",
                    schema=DomainSchema(
                        entities=["客户", "订单", "问题", "客服", "处理结果"],
                        relations=["购买", "咨询", "投诉", "处理", "反馈"]
                    ),
                    bridge_mappings=[
                        BridgeMapping(bridge_key="bridge_product_sku", role="涉及商品"),
                        BridgeMapping(bridge_key="bridge_issue_type", role="问题分类")
                    ]
                ),
                DomainDefinition(
                    domain_name="售后政策",
                    description="退换货政策、保修条款、服务标准",
                    trigger_condition="文档包含政策、条款、服务标准",
                    schema=DomainSchema(
                        entities=["政策", "条款", "标准", "期限"],
                        relations=["规定", "适用", "要求", "限制"]
                    ),
                    bridge_mappings=[
                        BridgeMapping(bridge_key="bridge_issue_type", role="适用场景")
                    ]
                )
            ]
        )

    @staticmethod
    def get_medical_template() -> GraphConfig:
        """医疗问诊模板"""
        return GraphConfig(
            project_name="医疗问诊系统",
            industry="医疗",
            description="适用于医疗咨询、电子病历、诊疗记录管理",
            bridge_definitions=[
                BridgeDefinition(
                    name="疾病名称",
                    key="bridge_disease",
                    description="疾病或症状的标准名称",
                    examples=["高血压", "糖尿病", "感冒"],
                    is_enum_restricted=False
                ),
                BridgeDefinition(
                    name="症状",
                    key="bridge_symptom",
                    description="患者的症状描述",
                    examples=["发热", "咳嗽", "头痛"],
                    is_enum_restricted=False
                )
            ],
            domain_definitions=[
                DomainDefinition(
                    domain_name="医学文献",
                    description="医学教材、诊疗指南、研究论文",
                    trigger_condition="文档包含疾病定义、诊疗规范、医学研究",
                    schema=DomainSchema(
                        entities=["疾病", "药物", "治疗方法", "诊断标准"],
                        relations=["治疗", "诊断", "引起", "预防"]
                    ),
                    bridge_mappings=[
                        BridgeMapping(bridge_key="bridge_disease", role="疾病名称"),
                        BridgeMapping(bridge_key="bridge_symptom", role="临床表现")
                    ]
                ),
                DomainDefinition(
                    domain_name="电子病历",
                    description="患者病历、诊断记录、检查报告",
                    trigger_condition="文档包含患者信息、病情描述、诊断结果",
                    schema=DomainSchema(
                        entities=["患者", "病情", "检查", "诊断", "医生"],
                        relations=["患有", "检查", "诊断为", "开具"]
                    ),
                    bridge_mappings=[
                        BridgeMapping(bridge_key="bridge_disease", role="诊断疾病"),
                        BridgeMapping(bridge_key="bridge_symptom", role="主诉症状")
                    ]
                ),
                DomainDefinition(
                    domain_name="处方记录",
                    description="用药记录、处方单、治疗方案",
                    trigger_condition="文档包含药品名称、用法用量、治疗方案",
                    schema=DomainSchema(
                        entities=["药品", "用法", "剂量", "疗程"],
                        relations=["用于", "治疗", "配合", "禁忌"]
                    ),
                    bridge_mappings=[
                        BridgeMapping(bridge_key="bridge_disease", role="适应症")
                    ]
                )
            ]
        )

    @staticmethod
    def get_all_templates() -> Dict[str, GraphConfig]:
        """获取所有预置模板"""
        return {
            "legal": IndustryTemplate.get_legal_template(),
            "ecommerce": IndustryTemplate.get_ecommerce_template(),
            "medical": IndustryTemplate.get_medical_template()
        }
