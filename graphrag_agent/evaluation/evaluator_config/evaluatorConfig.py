from typing import Dict, Any, List, Optional

class EvaluatorConfig:
    """评估器配置管理类"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        初始化评估器配置
        
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict or {}
        
        # 设置默认值
        self._set_defaults()
    
    def _set_defaults(self):
        """设置默认配置"""
        defaults = {
            'save_dir': './evaluation_results',
            'save_metric_score': True,
            'save_intermediate_data': True,
            'metrics': [],
            'debug': True,
            'dataset_name': 'default'
        }
        
        # 只设置未指定的默认值
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def get(self, key: str, default=None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置项键名
            default: 默认值
            
        Returns:
            配置项的值
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        设置配置项
        
        Args:
            key: 配置项键名
            value: 配置项的值
        """
        self.config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """
        更新配置字典
        
        Args:
            config_dict: 要更新的配置项字典
        """
        self.config.update(config_dict)
    
    def get_metrics(self) -> List[str]:
        """
        获取配置的评估指标列表
        
        Returns:
            List[str]: 评估指标列表
        """
        return [metric.lower() for metric in self.config.get('metrics', [])]
    
    def is_debug_enabled(self) -> bool:
        """
        判断是否开启调试模式
        
        Returns:
            bool: 是否开启调试模式
        """
        return self.config.get('debug', False)
    
    def get_save_dir(self) -> str:
        """
        获取评估结果保存目录
        
        Returns:
            str: 保存目录路径
        """
        return self.config.get('save_dir', './evaluation_results')
    
    def get_agent(self, agent_type: str) -> Optional[Any]:
        """
        获取指定类型的Agent实例
        
        Args:
            agent_type: Agent类型，可选值: naive, hybrid, graph, deep
            
        Returns:
            Any: Agent实例或None
        """
        agent_key = f"{agent_type}_agent"
        return self.config.get(agent_key)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return self.config.copy()