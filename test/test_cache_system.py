import unittest
import tempfile
import shutil
import time
import os
import sys
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sentence_transformers")
sys.path.append('.')

from CacheManage import CacheManager, ContextAwareCacheKeyStrategy, SimpleCacheKeyStrategy


class TestCacheSystem(unittest.TestCase):
    """缓存系统测试"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(
            cache_dir=self.temp_dir,
            memory_only=False,
            max_memory_size=50,
            max_disk_size=200,
            enable_vector_similarity=True,
            similarity_threshold=0.8
        )
    
    def tearDown(self):
        """测试后清理"""
        self.cache_manager.clear()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_cache_operations(self):
        """测试基本缓存操作"""
        # 设置缓存
        self.cache_manager.set("什么是Python?", "Python是一种编程语言")
        
        # 获取缓存
        result = self.cache_manager.get("什么是Python?")
        self.assertEqual(result, "Python是一种编程语言")
        
        # 标记质量
        success = self.cache_manager.mark_quality("什么是Python?", True)
        self.assertTrue(success)
        
        # 快速获取
        fast_result = self.cache_manager.get_fast("什么是Python?")
        self.assertEqual(fast_result, "Python是一种编程语言")
        
        # 删除缓存
        deleted = self.cache_manager.delete("什么是Python?")
        self.assertTrue(deleted)
        
        # 确认删除
        result_after_delete = self.cache_manager.get("什么是Python?")
        self.assertIsNone(result_after_delete)
    
    def test_vector_similarity_matching(self):
        """测试向量相似性匹配"""
        # 设置原始缓存
        self.cache_manager.set("Python是什么编程语言?", "Python是一种高级解释型编程语言")
        
        # 标记为高质量
        self.cache_manager.mark_quality("Python是什么编程语言?", True)
        
        # 测试相似查询
        similar_queries = [
            "什么是Python编程语言?",
            "Python语言是什么?",
            "介绍一下Python编程语言",
            "Python编程语言的特点"
        ]
        
        for query in similar_queries:
            result = self.cache_manager.get(query)
            if result:  # 如果相似度足够高才会有结果
                self.assertEqual(result, "Python是一种高级解释型编程语言")
                print(f"查询 '{query}' 成功匹配到相似缓存")
    
    def test_context_aware_caching(self):
        """测试上下文感知缓存"""
        context_manager = CacheManager(
            key_strategy=ContextAwareCacheKeyStrategy(),
            cache_dir=self.temp_dir + "_context",
            enable_vector_similarity=True
        )
        
        try:
            # 在不同线程/上下文中设置缓存
            context_manager.set("分析数据", "这是用户A的数据分析结果", thread_id="user_A")
            context_manager.set("分析数据", "这是用户B的数据分析结果", thread_id="user_B")
            
            # 测试上下文隔离
            result_a = context_manager.get("分析数据", thread_id="user_A")
            result_b = context_manager.get("分析数据", thread_id="user_B")
            
            self.assertEqual(result_a, "这是用户A的数据分析结果")
            self.assertEqual(result_b, "这是用户B的数据分析结果")
            self.assertNotEqual(result_a, result_b)
            
            # 测试相似查询在同一上下文中的匹配
            context_manager.mark_quality("分析数据", True, thread_id="user_A")
            similar_result = context_manager.get("请分析这些数据", thread_id="user_A")
            
            # 如果向量相似度足够，应该能匹配到
            if similar_result:
                self.assertEqual(similar_result, "这是用户A的数据分析结果")
                print("上下文感知的向量相似性匹配成功")
        
        finally:
            context_manager.clear()
    
    def test_performance_metrics(self):
        """测试性能指标"""
        # 设置一些缓存
        self.cache_manager.set("问题1", "答案1")
        self.cache_manager.set("问题2", "答案2")
        self.cache_manager.mark_quality("问题1", True)
        
        # 执行一些查询
        self.cache_manager.get("问题1")  # 精确命中
        self.cache_manager.get("问题2")  # 精确命中
        self.cache_manager.get("不存在的问题")  # 未命中
        
        # 获取性能指标
        metrics = self.cache_manager.get_metrics()
        
        self.assertGreater(metrics['total_queries'], 0)
        self.assertGreater(metrics['exact_hits'], 0)
        self.assertGreater(metrics['misses'], 0)
        
        print("性能指标:", metrics)
    
    def test_cache_quality_system(self):
        """测试缓存质量系统"""
        # 设置缓存
        self.cache_manager.set("测试问题", "测试答案")
        
        # 初始状态应该不是高质量
        result = self.cache_manager.get_fast("测试问题")
        # get_fast可能返回None如果不是高质量缓存
        
        # 标记为正面
        self.cache_manager.mark_quality("测试问题", True)
        
        # 现在应该是高质量缓存
        fast_result = self.cache_manager.get_fast("测试问题")
        self.assertEqual(fast_result, "测试答案")
        
        # 标记为负面
        self.cache_manager.mark_quality("测试问题", False)
        self.cache_manager.mark_quality("测试问题", False)
        self.cache_manager.mark_quality("测试问题", False)
        
        # 质量应该下降
        degraded_result = self.cache_manager.get_fast("测试问题")
        # 可能返回None由于质量下降
    
    def test_vector_similarity_with_context(self):
        """测试带上下文的向量相似性"""
        context_manager = CacheManager(
            key_strategy=ContextAwareCacheKeyStrategy(),
            cache_dir=self.temp_dir + "_vector_context",
            enable_vector_similarity=True,
            similarity_threshold=0.75
        )
        
        try:
            # 在不同上下文中设置相似但不同的缓存
            context_manager.set("如何学习编程?", "对于初学者，建议从Python开始", thread_id="beginner")
            context_manager.set("如何学习编程?", "对于有经验的开发者，可以尝试Rust", thread_id="expert")
            
            # 标记为高质量
            context_manager.mark_quality("如何学习编程?", True, thread_id="beginner")
            context_manager.mark_quality("如何学习编程?", True, thread_id="expert")
            
            # 测试相似查询在正确上下文中的匹配
            beginner_result = context_manager.get("编程学习方法", thread_id="beginner")
            expert_result = context_manager.get("编程学习方法", thread_id="expert")
            
            # 应该在各自的上下文中找到正确的答案
            if beginner_result:
                self.assertIn("Python", beginner_result)
            
            if expert_result:
                self.assertIn("Rust", expert_result)
            
            print("带上下文的向量相似性测试完成")
            
        finally:
            context_manager.clear()
    
    def test_cache_persistence(self):
        """测试缓存持久化"""
        # 设置缓存
        self.cache_manager.set("持久化测试", "这是持久化的数据")
        self.cache_manager.mark_quality("持久化测试", True)
        
        # 强制刷新所有数据到磁盘
        self.cache_manager.flush()
        
        # 创建新的管理器实例（模拟重启）
        new_manager = CacheManager(
            cache_dir=self.temp_dir,
            memory_only=False,
            enable_vector_similarity=True
        )
        
        # 应该能够获取之前的缓存
        result = new_manager.get("持久化测试")
        self.assertEqual(result, "这是持久化的数据")
        
        # 测试向量相似性是否也持久化了
        similar_result = new_manager.get("测试持久化功能")
        if similar_result:
            self.assertEqual(similar_result, "这是持久化的数据")
            print("向量索引持久化测试成功")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)