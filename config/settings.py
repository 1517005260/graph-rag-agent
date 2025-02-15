from pathlib import Path

# 知识图谱设置
theme="悟空传"
entity_types=["人物","妖怪","位置"]
relationship_types=["师徒", "师兄弟", "对抗", "对话", "态度", "故事地点", "其它"]

# 实体相似度
similarity_threshold = 0.9

# 社区算法：sllpa or leiden
community_algorithm = 'sllpa'

# 构建路径
BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / 'files'

# 文本分块
CHUNK_SIZE=300
OVERLAP=50

# 回答方式
response_type="多个段落"