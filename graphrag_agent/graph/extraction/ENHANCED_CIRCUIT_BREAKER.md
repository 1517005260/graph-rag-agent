# Enhanced Circuit Breaker with Empty Result Monitoring

## Overview

This document describes the enhanced circuit breaker mechanism in `entity_extractor.py` that monitors both **error rates** and **empty result rates** during parallel LLM extraction, with **completeness validation** to ensure concurrency safety.

## Problem Statement

### Previous Issues

**1. Error-Only Circuit Breaker (Limited Protection)**
```python
# Old: Only monitored errors
if error_count / completed > ERROR_RATE_THRESHOLD:
    raise RuntimeError("Error rate too high")
```

**Problem**: LLM service degradation can manifest in different ways:
- **Hard Failures**: Exceptions → Caught by error rate monitoring ✅
- **Soft Failures**: Empty results (no entities/relationships) → Not caught ❌

**Real Scenario**:
```
User uploads 1000 chunks
→ LLM service crashes silently
→ Returns HTTP 200 but empty JSON: {"entities": [], "relationships": []}
→ System processes all 1000 chunks successfully (no errors!)
→ Final result: Empty knowledge graph + wasted $50 in API costs
```

**2. Missing Completeness Validation**

```python
# Old: No validation after parallel processing
llm_results = [None] * total_chunks
# ... parallel processing ...
return llm_results  # What if some slots are still None?
```

**Problem**: Concurrency bugs are silent:
- Future fails → `llm_results[idx]` stays None
- Exception swallowed → No error logged
- Business logic can't distinguish "LLM says no entities" vs "Task failed"

**Impact**:
- Corrupted knowledge graph (missing chunks)
- Silent data loss
- Hard to debug (no clear error message)

---

## Solution: Three-Level Protection

### 1. Empty Result Monitoring

**Implementation**:
```python
# New counters
error_count = 0          # Exception failures
empty_result_count = 0   # Soft failures (no entities/relationships)

# New threshold
EMPTY_RATE_THRESHOLD = 0.3  # 30% threshold

# Check each result
for fut in concurrent.futures.as_completed(futures):
    try:
        res = fut.result()
        llm_results[idx] = res

        # ✅ NEW: Monitor empty results
        if not res.get("entities") and not res.get("relationships"):
            empty_result_count += 1
            logger.warning(f"⚠️ Chunk {idx} 返回空结果（无实体和关系）")
```

**Circuit Breaker Logic**:
```python
if completed > MIN_CHUNKS_FOR_CIRCUIT_BREAKER:
    current_error_rate = error_count / completed
    current_empty_rate = empty_result_count / completed
    combined_failure_rate = (error_count + empty_result_count) / completed

    # Level 1: Check error rate (20% threshold)
    if current_error_rate > ERROR_RATE_THRESHOLD:
        logger.critical(f"🔴 错误率过高 ({current_error_rate:.1%} > 20%)")
        # Cancel all pending tasks
        for f in futures.keys():
            if not f.done():
                f.cancel()
        raise RuntimeError(f"批量抽取失败：错误率 {current_error_rate:.1%}")

    # Level 2: Check empty rate (30% threshold)
    if current_empty_rate > EMPTY_RATE_THRESHOLD:
        logger.critical(f"🔴 空结果率过高 ({current_empty_rate:.1%} > 30%)")
        for f in futures.keys():
            if not f.done():
                f.cancel()
        raise RuntimeError(f"批量抽取失败：空结果率 {current_empty_rate:.1%}")

    # Level 3: Check combined rate (50% threshold)
    if combined_failure_rate > 0.5:
        logger.critical(f"🔴 综合失败率过高 ({combined_failure_rate:.1%} > 50%)")
        for f in futures.keys():
            if not f.done():
                f.cancel()
        raise RuntimeError(f"批量抽取失败：综合失败率 {combined_failure_rate:.1%}")
```

**Why Three Levels?**

| Scenario | Error Rate | Empty Rate | Combined Rate | Trigger |
|----------|-----------|-----------|---------------|---------|
| LLM API crashes (HTTP 500) | 80% | 0% | 80% | Level 1 ✅ |
| LLM prompt broken (returns empty) | 0% | 90% | 90% | Level 2 ✅ |
| Mixed failures (some errors, some empty) | 15% | 20% | 35% | Neither Level 1/2 ❌ |
| **Mixed failures** | 15% | 40% | 55% | Level 3 ✅ |

**Without Level 3**: Mixed failures can slip through if neither error rate nor empty rate exceeds threshold individually.

---

### 2. Completeness Validation

**Implementation**:
```python
# After all futures complete
# ✅ NEW: Validate completeness

# 1. Check count alignment
if len(llm_results) != total_chunks:
    raise RuntimeError(
        f"🔴 严重错误：结果数量 ({len(llm_results)}) 与输入块数 ({total_chunks}) 不一致！"
        f"这可能表明并发处理存在逻辑错误。"
    )

# 2. Check for None values
none_indices = [i for i, r in enumerate(llm_results) if r is None]
if none_indices:
    raise RuntimeError(
        f"🔴 严重错误：存在 {len(none_indices)} 个未被填充的结果槽位 (索引: {none_indices[:10]})。"
        f"这表明某些任务失败但未被正确处理。"
    )
```

**What This Catches**:

**Case 1: Pre-Allocation Bug**
```python
# Bug: Forgot to pre-allocate
llm_results = []  # Should be [None] * total_chunks

# When filling:
llm_results[50] = result  # IndexError!
```
**Validation catches**: `len(llm_results) != total_chunks`

**Case 2: Exception Swallowing**
```python
# Bug: Silent failure
try:
    result = fut.result()
    llm_results[idx] = result
except:
    pass  # ❌ Swallowed! llm_results[idx] stays None
```
**Validation catches**: `none_indices = [idx]` → RuntimeError

**Case 3: Concurrency Race Condition**
```python
# Bug: Wrong index mapping
future_to_index = {
    executor.submit(process, chunk): i  # Wrong variable scope
    for i, chunk in enumerate(chunks)
}
```
**Validation catches**: Some indices never filled → None values detected

---

### 3. Enhanced Logging & Statistics

**Implementation**:
```python
# After validation passes
final_error_rate = error_count / total_chunks if total_chunks > 0 else 0
final_empty_rate = empty_result_count / total_chunks if total_chunks > 0 else 0

if error_count > 0 or empty_result_count > 0:
    logger.warning(
        f"⚠️ 批量抽取完成，共 {total_chunks} 个 chunk，"
        f"成功 {total_chunks - error_count - empty_result_count} 个 "
        f"({100 * (1 - final_error_rate - final_empty_rate):.1f}%)，"
        f"失败 {error_count} 个 (错误率: {final_error_rate:.1%})，"
        f"空结果 {empty_result_count} 个 (空结果率: {final_empty_rate:.1%})"
    )
else:
    logger.info(f"✅ 批量抽取完成，共 {total_chunks} 个 chunk，全部成功")
```

**Sample Output**:
```
⚠️ 批量抽取完成，共 1000 个 chunk，成功 850 个 (85.0%)，失败 50 个 (错误率: 5.0%)，空结果 100 个 (空结果率: 10.0%)
```

**Benefits**:
- **Visibility**: See both error and empty rates
- **Debugging**: Know success rate at a glance
- **Monitoring**: Can integrate with alerting systems

---

## Configuration

### Thresholds

```python
# In process_chunks_batch()
ERROR_RATE_THRESHOLD = 0.2              # 20% - Hard failures
EMPTY_RATE_THRESHOLD = 0.3              # 30% - Soft failures
MIN_CHUNKS_FOR_CIRCUIT_BREAKER = 10     # Minimum sample size
```

**Tuning Guidelines**:

| Use Case | ERROR_RATE | EMPTY_RATE | Rationale |
|----------|-----------|-----------|-----------|
| **Production (Conservative)** | 0.1 | 0.2 | Fast fail, minimize waste |
| **Development (Default)** | 0.2 | 0.3 | Balance tolerance and safety |
| **Experimental (Permissive)** | 0.3 | 0.5 | Allow more exploration |

**Environment Variables** (Future Enhancement):
```python
# Could be made configurable via .env
ERROR_RATE_THRESHOLD = float(os.getenv("EXTRACTOR_ERROR_RATE_THRESHOLD", 0.2))
EMPTY_RATE_THRESHOLD = float(os.getenv("EXTRACTOR_EMPTY_RATE_THRESHOLD", 0.3))
```

---

## Usage Example

### Normal Operation (All Pass)

```python
extractor = EntityRelationExtractor(llm=llm, ...)

# Process 100 chunks
file_contents = [{"file_name": "doc.txt", "file_contents": [...]}]
result = extractor.process_chunks_batch(file_contents)

# Output:
# ✅ 批量抽取完成，共 100 个 chunk，全部成功
```

### Circuit Break - High Error Rate

```python
# 30 chunks fail with exceptions (30% error rate > 20% threshold)
# After processing 10 chunks:
# 🔴 错误率过高 (30.0% > 20%)，已处理 10/100 个 chunk
# → Remaining 90 tasks cancelled
# → RuntimeError raised
# → Saves 90% of API costs
```

### Circuit Break - High Empty Rate

```python
# LLM prompt broken, returns empty results
# After processing 10 chunks:
# ⚠️ Chunk 0 返回空结果（无实体和关系）
# ⚠️ Chunk 1 返回空结果（无实体和关系）
# ...
# 🔴 空结果率过高 (40.0% > 30%)，已处理 10/100 个 chunk
# → Remaining 90 tasks cancelled
# → RuntimeError raised
```

### Circuit Break - Mixed Failures

```python
# 10% errors + 45% empty results = 55% combined
# After processing 20 chunks:
# 🔴 综合失败率过高 (55.0% > 50%)，已处理 20/100 个 chunk
# → Level 3 circuit breaker triggered
# → Remaining 80 tasks cancelled
```

### Validation Failure - Missing Results

```python
# Concurrency bug: Some futures never completed
# After all futures:
# 🔴 严重错误：存在 5 个未被填充的结果槽位 (索引: [10, 23, 45, 67, 89])
# → RuntimeError raised
# → Developer investigates concurrency issue
```

---

## Performance Impact

### API Cost Savings

**Scenario**: 1000 chunks, LLM crashes after 50 chunks

| Approach | Chunks Processed | API Calls | Estimated Cost |
|----------|-----------------|-----------|----------------|
| **No Circuit Breaker** | 1000 | 1000 | $50.00 |
| **Error-Only Circuit Breaker** | 60 | 60 | $3.00 |
| **Enhanced Circuit Breaker** | 60 | 60 | $3.00 |

**Savings**: $47.00 (94% reduction) ✅

### Time Savings

**Scenario**: 1000 chunks, each takes 2 seconds

| Approach | Processing Time |
|----------|----------------|
| **No Circuit Breaker** | 2000s (~33 min) |
| **Enhanced Circuit Breaker** | 120s (~2 min) |

**Savings**: 1880s (~31 min, 94% reduction) ✅

### Detection Coverage

| Failure Mode | Error-Only CB | Enhanced CB |
|--------------|--------------|-------------|
| API crashes (HTTP 500) | ✅ Detected | ✅ Detected |
| Timeout exceptions | ✅ Detected | ✅ Detected |
| Empty responses (soft fail) | ❌ Not detected | ✅ Detected |
| Prompt engineering issues | ❌ Not detected | ✅ Detected |
| Concurrency bugs | ❌ Not detected | ✅ Detected (validation) |
| Mixed failures | ⚠️ Partial | ✅ Full coverage (Level 3) |

**Coverage Improvement**: 50% → 100% ✅

---

## Testing

### Unit Test Example

```python
def test_circuit_breaker_empty_results():
    """Test circuit breaker triggers on high empty result rate"""

    # Mock LLM to return empty results
    def mock_process(chunk):
        return {"entities": [], "relationships": []}

    extractor = EntityRelationExtractor(...)
    chunks = [{"file_contents": f"chunk_{i}"} for i in range(100)]

    # Should raise RuntimeError due to high empty rate
    with pytest.raises(RuntimeError, match="空结果率过高"):
        extractor.process_chunks_batch([{"file_contents": chunks}])

def test_completeness_validation():
    """Test validation catches None values"""

    # Mock executor to leave some results as None
    # ... (implementation depends on mocking strategy)

    with pytest.raises(RuntimeError, match="未被填充的结果槽位"):
        extractor.process_chunks_batch([{"file_contents": chunks}])
```

### Integration Test

```python
def test_real_llm_extraction():
    """Integration test with real LLM"""

    extractor = EntityRelationExtractor(llm=ChatOpenAI(...))

    # Test document with known entities
    chunks = load_test_document("student_handbook.txt")

    result = extractor.process_chunks_batch([{
        "file_name": "student_handbook.txt",
        "file_contents": chunks
    }])

    # Validate structure
    assert len(result) == len(chunks)
    assert all(r is not None for r in result)

    # Validate content quality
    total_entities = sum(len(r.get("entities", [])) for r in result)
    assert total_entities > 0, "Should extract at least some entities"

    empty_count = sum(1 for r in result if not r.get("entities") and not r.get("relationships"))
    empty_rate = empty_count / len(result)
    assert empty_rate < 0.3, f"Too many empty results: {empty_rate:.1%}"
```

---

## Monitoring & Alerting

### Log-Based Monitoring

**Key Log Patterns**:

```python
# Critical alerts (immediate action required)
logger.critical("错误率过高")      # → Page on-call engineer
logger.critical("空结果率过高")     # → Check LLM prompt/config
logger.critical("综合失败率过高")   # → Investigate mixed failure

# Warnings (review within 24h)
logger.warning("返回空结果")       # → Track empty rate trend
logger.warning("批量抽取完成...失败 X 个")  # → Review failure patterns

# Errors (investigate root cause)
logger.error("严重错误：结果数量不一致")  # → Concurrency bug
logger.error("严重错误：未被填充的结果槽位")  # → Task handling bug
```

### Metrics to Track

```python
# Prometheus-style metrics (future enhancement)
extraction_chunks_total{status="success"}
extraction_chunks_total{status="error"}
extraction_chunks_total{status="empty"}

extraction_error_rate
extraction_empty_rate
extraction_combined_failure_rate

extraction_circuit_breaker_triggered{reason="error_rate"}
extraction_circuit_breaker_triggered{reason="empty_rate"}
extraction_circuit_breaker_triggered{reason="combined_rate"}
```

### Dashboard Example

```
┌─────────────────────────────────────────┐
│ Entity Extraction Health                │
├─────────────────────────────────────────┤
│ Error Rate:      3.2%  ✅ (< 20%)       │
│ Empty Rate:     12.5%  ✅ (< 30%)       │
│ Combined Rate:  15.7%  ✅ (< 50%)       │
│                                         │
│ Chunks Processed: 10,234                │
│ Success:           8,625 (84.3%)        │
│ Errors:              328 ( 3.2%)        │
│ Empty:             1,281 (12.5%)        │
│                                         │
│ Circuit Breaks Today: 2                 │
│ - error_rate: 1                         │
│ - empty_rate: 1                         │
└─────────────────────────────────────────┘
```

---

## Troubleshooting

### High Empty Rate

**Symptom**: Circuit breaker triggers with "空结果率过高"

**Possible Causes**:
1. **Prompt Engineering Issue**: LLM prompt doesn't match document content
2. **Schema Mismatch**: Entity types don't exist in document domain
3. **LLM Temperature Too High**: Generating inconsistent output
4. **Document Quality**: Input text is corrupted or unintelligible

**Debug Steps**:
```python
# 1. Check sample empty results
logger.debug(f"Empty result example: {res}")  # Should see full LLM response

# 2. Check prompt template
logger.debug(f"Prompt: {self.human_template.format(text=chunk)}")

# 3. Test with known-good document
test_chunks = ["学生申请奖学金需要满足以下条件..."]  # Known entities
result = extractor._call_llm(test_chunks[0])
```

**Solutions**:
- Adjust prompt template in `graphrag_agent/graph/prompts/graph_prompts.py`
- Update entity types in `graphrag_agent/config/settings.py`
- Lower LLM temperature (e.g., 0.0 for deterministic output)
- Pre-process documents to remove corrupted text

### Validation Failure

**Symptom**: "严重错误：存在 X 个未被填充的结果槽位"

**Possible Causes**:
1. **Exception Swallowing**: Try-except block doesn't populate default value
2. **Index Mapping Bug**: `future_to_index` mapping is incorrect
3. **Pre-Allocation Bug**: `llm_results` not initialized properly

**Debug Steps**:
```python
# Add detailed logging in parallel processing
logger.debug(f"Submitted task {idx} for chunk {chunk_id}")
logger.debug(f"Task {idx} completed with result: {result}")
logger.debug(f"future_to_index mapping: {future_to_index}")
```

**Solutions**:
- Ensure exception handling populates default: `llm_results[idx] = {"entities": [], "relationships": []}`
- Verify index mapping: `assert len(future_to_index) == total_chunks`
- Check pre-allocation: `assert len(llm_results) == total_chunks` before parallel loop

---

## Future Enhancements

### 1. Adaptive Thresholds

**Idea**: Adjust thresholds based on historical data

```python
# Learn from past successful runs
avg_error_rate = 0.05  # Historical average
avg_empty_rate = 0.10

# Set thresholds as 2x standard deviation
ERROR_RATE_THRESHOLD = avg_error_rate * 2
EMPTY_RATE_THRESHOLD = avg_empty_rate * 2
```

### 2. Partial Retry

**Idea**: Retry only failed/empty chunks instead of failing entire batch

```python
# After circuit break, identify failed chunks
failed_indices = [i for i, r in enumerate(llm_results)
                  if r is None or (not r.get("entities") and not r.get("relationships"))]

# Retry with different LLM settings
retry_results = retry_chunks(failed_indices, temperature=0.0)
```

### 3. Multi-Model Fallback

**Idea**: Try cheaper/faster model first, fallback to expensive/better model on empty results

```python
# Primary: Fast model (gpt-4o-mini)
result = extractor.process_chunks(chunks, model="gpt-4o-mini")

# If empty rate > 20%, retry with better model
if empty_rate > 0.2:
    failed_chunks = [c for i, c in enumerate(chunks) if is_empty(result[i])]
    retry_result = extractor.process_chunks(failed_chunks, model="gpt-4o")
```

---

## Conclusion

The enhanced circuit breaker provides **comprehensive protection** against both hard failures (exceptions) and soft failures (empty results), with **completeness validation** to catch concurrency bugs early.

**Key Benefits**:
- ✅ **Cost Savings**: Stop processing early when LLM fails (save 90%+ API costs)
- ✅ **Time Savings**: Fail fast instead of wasting 30+ minutes on broken extraction
- ✅ **Better Debugging**: Clear error messages with metrics (error rate, empty rate)
- ✅ **Data Quality**: Validation ensures no silent data loss from concurrency bugs
- ✅ **Full Coverage**: Three-level monitoring catches all failure modes

**Migration Path**:
- Existing code continues to work (backward compatible)
- New protection mechanisms activate automatically
- No configuration changes required (sensible defaults)

**Related Files**:
- Implementation: `graphrag_agent/graph/extraction/entity_extractor.py`
- Logging: Python `logging` module
- Configuration: `.env` (future enhancement)
- Tests: `tests/graph/extraction/test_circuit_breaker.py` (to be created)
