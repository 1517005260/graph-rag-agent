from typing import List, Dict, Any
from collections import defaultdict
import logging

def num_tokens_from_string(text: str) -> int:
    """
    估算文本字符串中的token数量
    
    参数:
        text: 文本字符串
        
    返回:
        int: 估计的token数
    """
    try:
        from model.get_models import count_tokens
        return count_tokens(text)
    except:
        # 简单备用
        return len(text) // 4

def kb_prompt(kbinfos: Dict[str, List[Dict[str, Any]]], max_tokens: int = 4096) -> List[str]:
    """
    将知识库信息格式化为结构化提示
    
    参数:
        kbinfos: 包含chunks和文档聚合的字典
        max_tokens: 结果提示的最大token数
        
    返回:
        List[str]: 格式化的信息块列表
    """
    # 从chunks中提取content_with_weight
    knowledges = []
    for ck in kbinfos.get("chunks", []):
        content = ck.get("content_with_weight", ck.get("text", ""))
        if content:
            knowledges.append(content)
    
    # 限制总token数
    used_token_count = 0
    chunks_num = 0
    for i, c in enumerate(knowledges):
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]
            logging.warning(f"未将所有检索结果放入提示: {i+1}/{len(knowledges)}")
            break
    
    # 获取文档信息
    doc_aggs = kbinfos.get("doc_aggs", [])
    docs = {d.get("doc_id", ""): d for d in doc_aggs}
    
    # 按文档分组chunks
    doc2chunks = defaultdict(lambda: {"chunks": [], "meta": {}})
    for i, ck in enumerate(kbinfos.get("chunks", [])[:chunks_num]):
        # 获取文档名称或ID
        doc_id = ck.get("doc_id", ck.get("chunk_id", "unknown").split("_")[0] if "_" in ck.get("chunk_id", "") else "unknown")
        doc_name = doc_id
        
        # 如果有URL则添加
        url_prefix = f"URL: {ck['url']}\n" if "url" in ck else ""
        
        # 获取内容
        content = ck.get("content_with_weight", ck.get("text", ""))
        
        # 将chunk添加到文档组
        doc2chunks[doc_name]["chunks"].append(f"{url_prefix}ID: {i}\n{content}")
        
        # 如果有元数据则添加
        if doc_id in docs:
            doc2chunks[doc_name]["meta"] = {
                "title": docs[doc_id].get("title", doc_id),
                "type": docs[doc_id].get("type", "unknown")
            }
    
    # 格式化最终知识块
    formatted_knowledges = []
    for doc_name, cks_meta in doc2chunks.items():
        txt = f"\nDocument: {doc_name} \n"
        
        # 添加元数据
        for k, v in cks_meta["meta"].items():
            txt += f"{k}: {v}\n"
            
        txt += "Relevant fragments as following:\n"
        
        # 添加chunk内容
        for chunk in cks_meta["chunks"]:
            txt += f"{chunk}\n"
            
        formatted_knowledges.append(txt)
    
    # 如果没有找到chunks
    if not formatted_knowledges:
        return ["在知识库中未找到相关信息。"]
        
    return formatted_knowledges