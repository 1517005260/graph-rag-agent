from graphrag_agent.config.settings import examples as eg

# API服务器地址
API_URL = "http://localhost:8000"

# 示例问题
examples = eg

# 知识图谱颜色方案
KG_COLOR_PALETTE = [
    "#4285F4",  # 谷歌蓝
    "#EA4335",  # 谷歌红
    "#FBBC05",  # 谷歌黄
    "#34A853",  # 谷歌绿
    "#7B1FA2",  # 紫色
    "#0097A7",  # 青色
    "#FF6D00",  # 橙色
    "#757575",  # 灰色
    "#607D8B",  # 蓝灰色
    "#C2185B"   # 粉色
]

# 特殊节点类型的颜色映射
NODE_TYPE_COLORS = {
    "Center": "#F0B2F4",     # 中心/源节点 -紫色
    "Source": "#4285F4",     # 源节点 - 蓝色
    "Target": "#EA4335",     # 目标节点 - 红色
    "Common": "#34A853",     # 共同邻居 - 绿色
    "Level1": "#0097A7",     # 一级关联 - 青色
    "Level2": "#FF6D00",     # 二级关联 - 橙色
}

# 知识图谱默认设置
DEFAULT_KG_SETTINGS = {
    "physics_enabled": True,
    "node_size": 25,
    "edge_width": 2,
    "spring_length": 150,
    "gravity": -5000
}