# NDCG比论文基准低的原因分析

## 当前实现的问题

### 1. IDCG计算可能有问题 ⚠️

**当前代码**：
```python
idcg = 0.0
for no in range(recall):  # ⚠️ 基于实际命中的item数
    idcg += 1.0 / math.log(no + 2, 2)
```

**问题**：IDCG应该基于**真实标签的总数**，而不是**实际命中的item数**！

**正确的计算**应该是：
```python
idcg = 0.0
num_relevant = len(iid_list)  # 真实标签的总数
for no in range(min(num_relevant, topN)):  # 理想情况下，所有真实标签都在前面
    idcg += 1.0 / math.log(no + 2, 2)
```

**影响**：
- 如果真实标签有10个，但只命中了2个
- 当前IDCG = 1.0 + 0.63 = 1.63（基于2个）
- 正确IDCG应该基于10个计算 ≈ 4.96
- **NDCG会被严重低估**！

### 2. NDCG计算分母的问题

**当前实现**：
```python
if recall > 0:
    total_ndcg += dcg / idcg
else:
    total_ndcg += 0.0
ndcg = total_ndcg / total  # 除以所有样本
```

**问题**：如果hitrate很低（比如0.1），NDCG会被低估。

**可能原始实现用的是**：
```python
ndcg = total_ndcg / total_hitrate  # 只有有hit的样本的平均NDCG
```

## 论文基准对比

根据图片中的基准结果：
- **ComiRec-DR, NDCG@20**: 9.185%（即0.09185）
- **DNN, NDCG@20**: 7.670%（即0.07670）

如果你的NDCG远低于这个值，可能的原因：

### 1. IDCG计算错误（最可能）

如果IDCG基于实际命中数而不是真实标签数，会导致NDCG被严重低估。

### 2. 模型性能确实差

- Recall低 → 即使IDCG计算正确，DCG也会很低
- 排序质量差 → 真实item排在后面，DCG被折损函数惩罚

### 3. 计算方式不同

原始论文/代码可能：
- 只对有hit的样本计算NDCG，然后除以hitrate
- 或者使用了不同的IDCG计算方式

## 修复建议

### 方案1：修复IDCG计算（最重要）

```python
# 修复前
idcg = 0.0
for no in range(recall):  # ❌ 错误：基于命中数
    idcg += 1.0 / math.log(no + 2, 2)

# 修复后
idcg = 0.0
num_relevant = len(iid_list)  # 真实标签总数
for no in range(min(num_relevant, topN)):  # ✅ 正确：基于真实标签数
    idcg += 1.0 / math.log(no + 2, 2)
```

### 方案2：同时修复分母

根据原始实现，可能需要：
```python
# 如果原始实现用的是hitrate作为分母
ndcg = total_ndcg / total_hitrate if total_hitrate > 0 else 0.0
```

或者保持当前方式（除以total），但确保IDCG计算正确。

## 诊断代码

可以在评估函数中添加诊断信息：

```python
# 添加诊断
if recall > 0:
    dcg_val = dcg / idcg
    print(f"Sample: recall={recall}, num_relevant={len(iid_list)}, "
          f"idcg_based_on_hit={idcg:.4f}, "
          f"idcg_should_be={correct_idcg:.4f}, "
          f"ndcg={dcg_val:.4f}")
```

这样可以对比IDCG计算的差异。

