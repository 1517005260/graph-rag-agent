"""评估数据结构定义"""
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Tuple

from graphrag_agent.evaluation.preprocessing.text_cleaner import clean_thinking_process, clean_references
from graphrag_agent.evaluation.preprocessing.reference_extractor import extract_references_from_answer

class JsonSerializable:
    """可序列化为JSON的基类"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JsonSerializable':
        """从字典创建实例"""
        return cls(**data)

@dataclass
class AnswerEvaluationSample:
    """答案评估样本类，用于存储和更新答案评估数据"""
    
    question: str
    golden_answer: str
    system_answer: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    agent_type: str = ""  # naive, hybrid, graph, deep
    retrieved_entities: List[str] = field(default_factory=list)
    retrieved_relationships: List = field(default_factory=list)
    
    def update_system_answer(self, answer: str, agent_type: str = ""):
        """
        更新系统回答，自动清理引用数据和思考过程
        
        Args:
            answer: 原始系统回答
            agent_type: Agent类型
        """
        # 先清理思考过程，再清理引用数据
        cleaned_answer = clean_thinking_process(answer)
        cleaned_answer = clean_references(cleaned_answer)
        
        self.system_answer = cleaned_answer
        if agent_type:
            self.agent_type = agent_type
            
    def update_evaluation_score(self, metric: str, score: float):
        """更新评估分数"""
        self.scores[metric] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass
class AnswerEvaluationData:
    """答案评估数据类，用于管理多个答案评估样本"""
    
    samples: List[AnswerEvaluationSample] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> AnswerEvaluationSample:
        return self.samples[idx]
    
    def append(self, sample: AnswerEvaluationSample):
        """添加评估样本"""
        self.samples.append(sample)
    
    @property
    def questions(self) -> List[str]:
        """获取所有问题"""
        return [sample.question for sample in self.samples]
    
    @property
    def golden_answers(self) -> List[str]:
        """获取所有标准答案"""
        return [sample.golden_answer for sample in self.samples]
    
    @property
    def system_answers(self) -> List[str]:
        """获取所有系统回答"""
        return [sample.system_answer for sample in self.samples]
    
    def save(self, path: str):
        """保存评估数据"""
        with open(path, "w", encoding='utf-8') as f:
            json.dump([sample.to_dict() for sample in self.samples], f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AnswerEvaluationData':
        """加载评估数据"""
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)
        
        data = cls()
        for sample_data in samples_data:
            sample = AnswerEvaluationSample(**sample_data)
            data.append(sample)
        
        return data

@dataclass
class RetrievalEvaluationSample:
    """检索评估样本类"""
    
    question: str
    system_answer: str = ""
    retrieved_entities: List[str] = field(default_factory=list)
    retrieved_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    referenced_entities: List[str] = field(default_factory=list)
    referenced_relationships: List = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    agent_type: str = ""  # naive, hybrid, graph, deep
    retrieval_time: float = 0.0
    retrieval_logs: Dict[str, Any] = field(default_factory=dict)
    entity_details: List[Dict[str, str]] = field(default_factory=list)
    enhanced_relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    
    def update_system_answer(self, answer: str, agent_type: str = ""):
        """更新系统回答并提取引用"""
        # 如果是deep agent，先清理思考过程
        if agent_type == "deep":
            answer = clean_thinking_process(answer)
            
        # 保存原始答案（包含引用数据）
        self.system_answer = answer
        
        if agent_type:
            self.agent_type = agent_type
                
        # 提取引用的实体和关系
        refs = extract_references_from_answer(answer)
        
        # 将提取的实体和关系ID存储为字符串列表
        self.referenced_entities = refs.get("entities", [])
        # 关系暂时存储为ID，后续在evaluation方法中再转换为三元组
        self.referenced_relationships = refs.get("relationships", [])
    
    def update_retrieval_data(self, entities: List[str], relationships: List[Tuple[str, str, str]]):
        """更新检索到的实体和关系"""
        self.retrieved_entities = entities
        self.retrieved_relationships = relationships
        
    def update_logs(self, logs: Dict[str, Any]):
        """更新检索日志"""
        self.retrieval_logs = logs
    
    def update_evaluation_score(self, metric: str, score: float):
        """更新评估分数"""
        self.scores[metric] = score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        
        # 处理关系元组（JSON序列化时需要转换为列表）
        result["retrieved_relationships"] = [list(rel) for rel in self.retrieved_relationships]
        # 处理enhanced_relationships字段
        if hasattr(self, 'enhanced_relationships') and self.enhanced_relationships:
            result["enhanced_relationships"] = [list(rel) for rel in self.enhanced_relationships]
        
        # 处理检索日志中可能存在的HumanMessage
        if "retrieval_logs" in result and isinstance(result["retrieval_logs"], dict):
            logs = result["retrieval_logs"]
            if "execution_log" in logs and isinstance(logs["execution_log"], list):
                for i, log in enumerate(logs["execution_log"]):
                    # 处理输入中可能的HumanMessage
                    if "input" in log and hasattr(log["input"], "__class__") and log["input"].__class__.__name__ == "HumanMessage":
                        logs["execution_log"][i]["input"] = str(log["input"])
                    # 处理输出中可能的HumanMessage或AIMessage
                    if "output" in log and hasattr(log["output"], "__class__") and log["output"].__class__.__name__ in ["HumanMessage", "AIMessage"]:
                        logs["execution_log"][i]["output"] = str(log["output"])
        
        return result

@dataclass
class RetrievalEvaluationData:
    """检索评估数据类"""
    
    samples: List[RetrievalEvaluationSample] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> RetrievalEvaluationSample:
        return self.samples[idx]
    
    def append(self, sample: RetrievalEvaluationSample):
        """添加评估样本"""
        self.samples.append(sample)
    
    @property
    def questions(self) -> List[str]:
        """获取所有问题"""
        return [sample.question for sample in self.samples]
    
    @property
    def system_answers(self) -> List[str]:
        """获取所有系统回答"""
        return [sample.system_answer for sample in self.samples]
    
    @property
    def retrieved_entities(self) -> List[List[str]]:
        """获取所有检索到的实体"""
        return [sample.retrieved_entities for sample in self.samples]
    
    @property
    def referenced_entities(self) -> List[List[str]]:
        """获取所有引用的实体"""
        return [sample.referenced_entities for sample in self.samples]
    
    @property
    def retrieved_relationships(self) -> List[List[Tuple[str, str, str]]]:
        """获取所有检索到的关系"""
        return [sample.retrieved_relationships for sample in self.samples]
    
    @property
    def referenced_relationships(self) -> List[List]:
        """获取所有引用的关系"""
        return [sample.referenced_relationships for sample in self.samples]
    
    def save(self, path: str):
        """保存评估数据"""
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    from langchain_core.messages import BaseMessage
                    if isinstance(obj, BaseMessage):
                        return str(obj)
                except ImportError:
                    pass
                return super().default(obj)
        
        with open(path, "w", encoding='utf-8') as f:
            samples_data = [sample.to_dict() for sample in self.samples]
            json.dump(samples_data, f, ensure_ascii=False, indent=2, cls=CustomEncoder)
    
    @classmethod
    def load(cls, path: str) -> 'RetrievalEvaluationData':
        """加载评估数据"""
        with open(path, "r", encoding='utf-8') as f:
            samples_data = json.load(f)
        
        data = cls()
        for sample_data in samples_data:
            # 转换关系格式（从列表到元组）
            if "retrieved_relationships" in sample_data:
                sample_data["retrieved_relationships"] = [tuple(rel) for rel in sample_data["retrieved_relationships"]]
            if "enhanced_relationships" in sample_data:
                sample_data["enhanced_relationships"] = [tuple(rel) for rel in sample_data["enhanced_relationships"]]
                
            sample = RetrievalEvaluationSample(**sample_data)
            data.append(sample)
        
        return data