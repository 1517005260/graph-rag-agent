import re
import json
from typing import Dict, Any, List, Optional

def extract_references_from_answer(answer: str) -> Dict[str, Any]:
    """
    从回答中提取引用数据，增强处理各种格式的能力
    
    Args:
        answer: AI生成的回答
        
    Returns:
        Dict: 包含entities, relationships, chunks等信息的字典
    """
    # 初始化结果
    result = {
        "entities": [],
        "relationships": [],
        "chunks": [],
        "reports": []
    }
    
    # 如果没有回答或引用数据部分，直接返回空结果
    if not answer or "引用数据" not in answer:
        return result
    
    try:
        # 先尝试提取完整的引用数据部分
        reference_section = extract_reference_section(answer)
        if not reference_section:
            return result
            
        # 尝试多种方式解析JSON格式的引用数据
        parsed_data = parse_json_data(reference_section)
        if parsed_data:
            # 处理实体
            entities = extract_entities_from_parsed(parsed_data)
            result["entities"].extend(entities)
            
            # 处理关系
            relationships = extract_relationships_from_parsed(parsed_data)
            result["relationships"].extend(relationships)
            
            # 处理文本块
            chunks = extract_chunks_from_parsed(parsed_data)
            result["chunks"].extend(chunks)

            # 处理报告
            reports = extract_reports_from_parsed(parsed_data)
            result["reports"].extend(reports)
        else:
            # 如果无法解析JSON，尝试直接从文本中提取
            result["entities"] = extract_entities_from_text(reference_section)
            result["relationships"] = extract_relationships_from_text(reference_section)
            result["chunks"] = extract_chunks_from_text(reference_section)
            result["reports"] = extract_reports_from_text(reference_section)
        
        # 验证和格式化提取的ID
        result["entities"] = validate_and_format_ids(result["entities"])
        result["relationships"] = validate_and_format_ids(result["relationships"])
        
        # 去重处理
        result["entities"] = list(set(result["entities"]))
        result["relationships"] = list(set(result["relationships"]))
        result["chunks"] = list(set(result["chunks"]))
        result["reports"] = list(set(result["reports"]))
        
        return result
    except Exception as e:
        print(f"提取引用数据时出错: {e}")
        return result

def validate_and_format_ids(ids_list: List) -> List[str]:
    """
    验证并格式化ID列表，处理不同格式的ID
    
    Args:
        ids_list: 包含各种格式ID的列表
        
    Returns:
        List[str]: 格式化后的ID列表
    """
    valid_ids = []
    for id_value in ids_list:
        # 跳过None和空值
        if id_value is None or id_value == "":
            continue
            
        # 尝试处理不同格式的ID
        if isinstance(id_value, (int, float)):
            valid_ids.append(str(int(id_value)))
        elif isinstance(id_value, str):
            # 如果是数字字符串，直接添加
            if id_value.isdigit() or id_value.lstrip('-').isdigit():
                valid_ids.append(id_value)
            # 如果看起来像是UUID或其他特殊ID格式(长字符串)，也添加
            elif len(id_value) > 10:
                valid_ids.append(id_value)
            # 其他非空字符串也添加
            elif id_value.strip():
                valid_ids.append(id_value)
    return valid_ids

def extract_reference_section(answer: str) -> str:
    """提取回答中的引用数据部分"""
    # 尝试多种引用数据标记格式
    patterns = [
        r'#{1,4}\s*引用数据[\s\S]*?(\{[\s\S]*?\})\s*$',  # #### 引用数据 {...}
        r'引用数据[：:]\s*(\{[\s\S]*?\})\s*$',           # 引用数据: {...}
        r'<引用数据>\s*(\{[\s\S]*?\})\s*</引用数据>',    # <引用数据> {...} </引用数据>
        r'引用[：:]\s*(\{[\s\S]*?\})\s*$',               # 引用: {...}
        r'参考[：:]\s*(\{[\s\S]*?\})\s*$',               # 参考: {...}
        r'数据[：:]\s*(\{[\s\S]*?\})\s*$',               # 数据: {...}
        r'(\{[\s\S]*?[\'"]*data[\'"]*[\s\S]*?\})\s*$'    # {...data...}
    ]
    
    for pattern in patterns:
        match = re.search(pattern, answer, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return ""

def parse_json_data(data_text: str) -> Optional[Dict]:
    """尝试多种方式解析JSON数据"""
    # 直接尝试解析
    try:
        parsed = json.loads(data_text)
        return parsed
    except:
        pass
    
    # 尝试修复常见JSON格式问题
    try:
        # 修复单引号问题
        fixed_text = data_text.replace("'", '"')
        parsed = json.loads(fixed_text)
        return parsed
    except:
        pass
    
    # 尝试提取data字段
    try:
        data_match = re.search(r'\{\s*["\']*data["\']*\s*:\s*(\{[\s\S]*?\})\s*\}', data_text, re.DOTALL)
        if data_match:
            data_content = data_match.group(1)
            # 修复单引号
            fixed_text = "{\"data\":" + data_content.replace("'", '"') + "}"
            parsed = json.loads(fixed_text)
            return parsed
    except:
        pass
    
    # 尝试将text包装成合法JSON
    try:
        # 去除非ASCII字符
        cleaned_text = ''.join(c for c in data_text if ord(c) < 128)
        # 替换所有单引号为双引号
        cleaned_text = cleaned_text.replace("'", '"')
        # 确保键名有双引号
        cleaned_text = re.sub(r'(\w+)(?=\s*:)', r'"\1"', cleaned_text)
        parsed = json.loads(cleaned_text)
        return parsed
    except:
        return None

def extract_entities_from_parsed(parsed_data: Dict) -> List[str]:
    """
    从解析后的数据中提取实体ID
    
    Args:
        parsed_data: 已解析的数据
    
    Returns:
        List[str]: 实体ID列表
    """
    entities = []
    
    # 处理嵌套的data结构
    if "data" in parsed_data and isinstance(parsed_data["data"], dict):
        parsed_data = parsed_data["data"]
    
    # 提取Entities字段的值
    entity_keys = ["Entities", "entities", "Entity", "entity"]
    for key in entity_keys:
        if key in parsed_data and parsed_data[key]:
            if isinstance(parsed_data[key], list):
                # 处理列表格式
                for item in parsed_data[key]:
                    if isinstance(item, (int, float)):
                        entities.append(str(int(item)))
                    elif isinstance(item, str):
                        entities.append(item)
                    elif isinstance(item, dict) and "id" in item:
                        # 处理{id: 123}格式
                        entities.append(str(item["id"]))
            elif isinstance(parsed_data[key], str):
                # 处理逗号分隔的字符串
                parts = parsed_data[key].split(",")
                for part in parts:
                    clean_part = part.strip()
                    if clean_part:
                        entities.append(clean_part)
            elif isinstance(parsed_data[key], dict):
                # 处理字典格式
                for k, v in parsed_data[key].items():
                    if isinstance(v, (int, str)):
                        entities.append(str(v))
    
    return entities

def extract_relationships_from_parsed(parsed_data: Dict) -> List[str]:
    """
    从解析后的数据中提取关系ID
    
    Args:
        parsed_data: 已解析的数据
    
    Returns:
        List[str]: 关系ID列表
    """
    relationships = []
    
    # 处理嵌套的data结构
    if "data" in parsed_data and isinstance(parsed_data["data"], dict):
        parsed_data = parsed_data["data"]
    
    # 提取关系ID的所有可能键
    rel_keys = [
        "Relationships", "relationships", "Relations", "relations", 
        "Relation", "relation", "Reports", "reports", "Report", "report"
    ]
    
    for key in rel_keys:
        if key in parsed_data and parsed_data[key]:
            if isinstance(parsed_data[key], list):
                # 处理列表格式
                for item in parsed_data[key]:
                    if isinstance(item, (int, float)):
                        relationships.append(str(int(item)))
                    elif isinstance(item, str):
                        relationships.append(item)
                    elif isinstance(item, dict) and "id" in item:
                        # 处理{id: 123}格式
                        relationships.append(str(item["id"]))
                    elif isinstance(item, tuple) or (isinstance(item, list) and len(item) >= 3):
                        # 处理三元组格式 (source, relation, target)
                        # 在这种情况下，我们可以提取关系ID或使用整个三元组
                        relationships.append(str(item))
            elif isinstance(parsed_data[key], str):
                # 处理逗号分隔的字符串
                parts = parsed_data[key].split(",")
                for part in parts:
                    clean_part = part.strip()
                    if clean_part:
                        relationships.append(clean_part)
            elif isinstance(parsed_data[key], dict):
                # 处理字典格式
                for k, v in parsed_data[key].items():
                    if isinstance(v, (int, str)):
                        relationships.append(str(v))
    
    return relationships

def extract_chunks_from_parsed(parsed_data: Dict) -> List[str]:
    """从解析后的数据中提取文本块ID"""
    chunks = []
    
    # 处理嵌套的data结构
    if "data" in parsed_data and isinstance(parsed_data["data"], dict):
        parsed_data = parsed_data["data"]
    
    # 提取Chunks字段的值
    chunk_keys = ["Chunks", "chunks", "Chunk", "chunk", "TextChunks", "textchunks"]
    for key in chunk_keys:
        if key in parsed_data and parsed_data[key]:
            if isinstance(parsed_data[key], list):
                # 处理字符串列表
                for item in parsed_data[key]:
                    if isinstance(item, str):
                        chunks.append(item)
            elif isinstance(parsed_data[key], str):
                # 如果是逗号分隔的字符串
                chunks.extend([c.strip() for c in parsed_data[key].split(",") if c.strip()])
    
    return chunks

def extract_reports_from_parsed(parsed_data: Dict) -> List[str]:
    """从解析后的数据中提取报告ID"""
    reports = []
    
    # 处理嵌套的data结构
    if "data" in parsed_data and isinstance(parsed_data["data"], dict):
        parsed_data = parsed_data["data"]
    
    # 提取Reports字段的值
    report_keys = ["Reports", "reports", "Report", "report"]
    for key in report_keys:
        if key in parsed_data and parsed_data[key]:
            if isinstance(parsed_data[key], list):
                for item in parsed_data[key]:
                    if isinstance(item, (int, str)):
                        reports.append(str(item))
            elif isinstance(parsed_data[key], str):
                reports.extend([r.strip() for r in parsed_data[key].split(",") if r.strip()])
    
    return reports

def extract_entities_from_text(text: str) -> List[str]:
    """直接从文本中提取实体ID"""
    # 尝试匹配实体ID部分
    entity_matches = re.search(r'[Ee]ntities\s*[=:]\s*\[(.*?)\]', text, re.DOTALL) or \
                    re.search(r'[Ee]ntities\s*[=:]\s*([\d\s,]+)', text, re.DOTALL)
    
    if entity_matches:
        entity_str = entity_matches.group(1).strip()
        # 提取数字
        return re.findall(r'\d+', entity_str)
    
    return []

def extract_relationships_from_text(text: str) -> List[str]:
    """直接从文本中提取关系ID"""
    # 尝试匹配关系ID部分
    rel_matches = re.search(r'[Rr]elationships\s*[=:]\s*\[(.*?)\]', text, re.DOTALL) or \
                re.search(r'[Rr]elationships\s*[=:]\s*([\d\s,]+)', text, re.DOTALL) or \
                re.search(r'[Rr]eports\s*[=:]\s*\[(.*?)\]', text, re.DOTALL) or \
                re.search(r'[Rr]eports\s*[=:]\s*([\d\s,]+)', text, re.DOTALL)
    
    if rel_matches:
        rel_str = rel_matches.group(1).strip()
        # 提取数字
        return re.findall(r'\d+', rel_str)
    
    return []

def extract_chunks_from_text(text: str) -> List[str]:
    """直接从文本中提取文本块ID"""
    # 尝试匹配文本块ID部分
    chunk_matches = re.search(r'[Cc]hunks\s*[=:]\s*\[(.*?)\]', text, re.DOTALL)
    
    if chunk_matches:
        chunk_str = chunk_matches.group(1).strip()
        # 提取引号中的内容
        return re.findall(r'[\'"]([^\'"]*)[\'"]', chunk_str)
    
    return []

def extract_reports_from_text(text: str) -> List[str]:
    """直接从文本中提取报告ID"""
    # 尝试匹配报告ID部分
    report_matches = re.search(r'[Rr]eports\s*[=:]\s*\[(.*?)\]', text, re.DOTALL) or \
                    re.search(r'[Rr]eports\s*[=:]\s*([\d\s,]+)', text, re.DOTALL)
    
    if report_matches:
        report_str = report_matches.group(1).strip()
        # 提取数字
        return re.findall(r'\d+', report_str)
    
    return []