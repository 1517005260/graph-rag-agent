import hanlp
import re
from typing import List, Tuple

from graphrag_agent.config.settings import CHUNK_SIZE, OVERLAP, MAX_TEXT_LENGTH

class ChineseTextChunker:
    """中文文本分块器，将长文本分割成带有重叠的文本块"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP, max_text_length: int = MAX_TEXT_LENGTH):
        """
        初始化分块器
        
        Args:
            chunk_size: 每个文本块的目标大小（tokens数量）
            overlap: 相邻文本块的重叠大小（tokens数量）
            max_text_length: HanLP处理的最大文本长度，超过此长度将进行预分割
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size必须大于overlap")
            
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_text_length = max_text_length
        self.tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        
    def process_files(self, file_contents: List[Tuple[str, str]]) -> List[Tuple[str, str, List[List[str]]]]:
        """
        处理多个文件的内容
        
        Args:
            file_contents: List of (filename, content) tuples
            
        Returns:
            List of (filename, content, chunks) tuples
        """
        results = []
        for filename, content in file_contents:
            chunks = self.chunk_text(content)
            results.append((filename, content, chunks))
        return results
    
    def _preprocess_large_text(self, text: str) -> List[str]:
        """
        预处理过大的文本，将其分割成较小的段落
        
        Args:
            text: 原始文本
            
        Returns:
            分割后的文本段落列表
        """
        if len(text) <= self.max_text_length:
            return [text]
        
        # 计算合适的段落大小（确保不超过最大长度，但也不要太小）
        target_segment_size = min(self.max_text_length, max(10000, self.max_text_length // 2))
        
        # 首先按段落分割
        paragraphs = text.split('\n\n')
        
        # 如果段落数量很少，尝试按单个换行符分割
        if len(paragraphs) < 5:
            paragraphs = text.split('\n')
        
        # 重新组合段落，确保每个段落不超过目标大小
        processed_segments = []
        current_segment = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # 如果当前段落本身就超长，需要进一步分割
            if len(para) > target_segment_size:
                # 先保存当前积累的内容
                if current_segment:
                    processed_segments.append(current_segment)
                    current_segment = ""
                
                # 分割超长段落
                split_paras = self._split_long_paragraph(para, target_segment_size)
                processed_segments.extend(split_paras)
                
            else:
                # 检查添加当前段落是否会超长
                if len(current_segment) + len(para) + 2 > target_segment_size:  # +2 for \n\n
                    if current_segment:
                        processed_segments.append(current_segment)
                    current_segment = para
                else:
                    if current_segment:
                        current_segment += "\n\n" + para
                    else:
                        current_segment = para
        
        # 添加最后的segment
        if current_segment:
            processed_segments.append(current_segment)
        
        return processed_segments
    
    def _split_long_paragraph(self, text: str, max_size: int) -> List[str]:
        """
        分割超长段落
        
        Args:
            text: 超长段落文本
            max_size: 最大分割大小
            
        Returns:
            分割后的段落列表
        """
        if len(text) <= max_size:
            return [text]
        
        # 按句子分割
        sentences = re.split(r'([。！？.!?])', text)
        
        # 重新组合句子和标点
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence.strip():
                combined_sentences.append(sentence + punctuation)
        
        # 如果没有找到句子边界，按固定长度分割
        if not combined_sentences:
            result = []
            for i in range(0, len(text), max_size):
                result.append(text[i:i + max_size])
            return result
        
        # 重新组合句子，确保不超过最大长度
        segments = []
        current_segment = ""
        
        for sentence in combined_sentences:
            # 如果单个句子就超长，强制分割
            if len(sentence) > max_size:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = ""
                
                # 按固定长度分割超长句子
                for i in range(0, len(sentence), max_size):
                    segments.append(sentence[i:i + max_size])
            else:
                # 检查添加当前句子是否会超长
                if len(current_segment) + len(sentence) > max_size:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment += sentence
        
        # 添加最后的segment
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _safe_tokenize(self, text: str) -> List[str]:
        """
        安全的分词方法，处理可能的异常
        
        Args:
            text: 要分词的文本
            
        Returns:
            分词结果列表
        """
        try:
            # 检查文本长度
            if len(text) > self.max_text_length:
                return list(text)
            
            tokens = self.tokenizer(text)
            return tokens if tokens else []
        except Exception:
            return list(text)
        
    def chunk_text(self, text: str) -> List[List[str]]:
        """
        将单个文本分割成块
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表，每个块是token列表
        """
        # 处理空文本或太短的文本
        if not text or len(text) < self.chunk_size / 10:
            tokens = self._safe_tokenize(text)
            return [tokens] if tokens else []
        
        # 预处理过大文本
        text_segments = self._preprocess_large_text(text)
        
        # 处理每个文本段落
        all_chunks = []
        for segment in text_segments:
            segment_chunks = self._chunk_single_segment(segment)
            all_chunks.extend(segment_chunks)
        
        return all_chunks
    
    def _chunk_single_segment(self, text: str) -> List[List[str]]:
        """
        处理单个文本段落的分块
        
        Args:
            text: 单个文本段落
            
        Returns:
            分块结果
        """
        if not text:
            return []
            
        # 先将整个文本分词
        all_tokens = self._safe_tokenize(text)
        if not all_tokens:
            return []
        
        chunks = []
        start_pos = 0
        
        while start_pos < len(all_tokens):
            # 确定当前块的结束位置
            end_pos = min(start_pos + self.chunk_size, len(all_tokens))
            
            # 如果不是最后一块，尝试在句子边界结束
            if end_pos < len(all_tokens):
                # 寻找句子结束位置
                sentence_end = self._find_next_sentence_end(all_tokens, end_pos)
                if sentence_end <= start_pos + self.chunk_size + 100:  # 允许略微超出
                    end_pos = sentence_end
            
            # 提取当前块
            chunk = all_tokens[start_pos:end_pos]
            if chunk:  # 确保块不为空
                chunks.append(chunk)
            
            # 计算下一块的起始位置（考虑重叠）
            if end_pos >= len(all_tokens):
                break
                
            # 寻找重叠的起始位置
            overlap_start = max(start_pos, end_pos - self.overlap)
            next_sentence_start = self._find_previous_sentence_end(all_tokens, overlap_start)
            
            # 如果找到合适的句子开始位置，使用它；否则使用计算的重叠位置
            if next_sentence_start > start_pos and next_sentence_start < end_pos:
                start_pos = next_sentence_start
            else:
                start_pos = overlap_start
                
            # 防止无限循环
            if start_pos >= end_pos:
                start_pos = end_pos
        
        return chunks
    
    def _is_sentence_end(self, token: str) -> bool:
        """判断token是否为句子结束符"""
        return token in ['。', '！', '？']
    
    def _find_next_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """从指定位置向后查找句子结束位置"""
        for i in range(start_pos, len(tokens)):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return len(tokens)
    
    def _find_previous_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        """从指定位置向前查找句子结束位置"""
        for i in range(start_pos - 1, -1, -1):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return 0
    
    def get_text_stats(self, text: str) -> dict:
        """
        获取文本统计信息
        
        Args:
            text: 输入文本
            
        Returns:
            包含文本统计信息的字典
        """
        stats = {
            'text_length': len(text),
            'needs_preprocessing': len(text) > self.max_text_length,
            'estimated_chunks': max(1, len(text) // self.chunk_size),
            'paragraphs': len(text.split('\n\n')),
            'lines': len(text.split('\n'))
        }
        
        if stats['needs_preprocessing']:
            segments = self._preprocess_large_text(text)
            stats['preprocessed_segments'] = len(segments)
            stats['max_segment_length'] = max(len(seg) for seg in segments) if segments else 0
            
        return stats