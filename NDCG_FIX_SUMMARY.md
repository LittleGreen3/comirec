# NDCG计算修复总结

## 发现的严重Bug

### Bug 1: IDCG计算错误 ⚠️ **最严重**

**原来的错误代码**：
```python
idcg = 0.0
for no in range(recall):  # ❌ 基于实际命中的item数
    idcg += 1.0 / math.log(no + 2, 2)
```

**问题**：
- IDCG应该表示"理想情况下所有相关item都在最前面时的DCG"
- 应该基于**真实标签的总数**计算，而不是**实际命中数**
- 这导致NDCG被严重低估！

**举例**：
- 真实标签有10个，但只命中了2个
- **错误IDCG**：基于2个计算 = 1.0 + 0.63 = 1.63
- **正确IDCG**：基于10个计算 ≈ 4.96
- **NDCG会被低估约3倍！**

**修复后的代码**：
```python
# 修复IDCG计算：应该基于真实标签总数，而不是实际命中数
idcg = 0.0
num_relevant = len(iid_list)  # 真实标签的总数
for no in range(min(num_relevant, topN)):  # 理想情况下，所有真实标签都在前面（最多topN个）
    idcg += 1.0 / math.log(no + 2, 2)
```

### Bug 2: NDCG分母问题（已修复）

**原来的代码**：
```python
if recall > 0:
    total_ndcg += dcg / idcg
# recall=0时，不累加（导致逻辑不一致）

ndcg = total_ndcg / total
```

**修复后**：
```python
if recall > 0:
    total_ndcg += dcg / idcg
else:
    total_ndcg += 0.0  # recall=0时，NDCG=0

ndcg = total_ndcg / total  # 所有样本的平均NDCG
```

## 修复对比

### 修复前
- IDCG基于实际命中数 → NDCG被严重低估
- recall=0的样本不参与NDCG计算 → 逻辑不一致

### 修复后
- IDCG基于真实标签总数 → NDCG计算正确
- 所有样本都参与NDCG计算（recall=0时为0）→ 逻辑一致

## 与论文基准的对比

根据原始论文（TensorFlow 1.14）：
- **ComiRec-DR, NDCG@20**: 9.185%（即0.09185）
- **DNN, NDCG@20**: 7.670%（即0.07670）

**修复IDCG后，NDCG应该会显著提升**，接近论文基准值。

## 为什么之前NDCG很低？

1. **IDCG计算错误**（最主要）：
   - 基于实际命中数而不是真实标签总数
   - 导致NDCG被低估2-5倍（取决于recall）

2. **模型性能本身可能较差**：
   - Recall低 → DCG低
   - 排序质量差 → 真实item排在后面，DCG被折损

3. **训练不充分**：
   - Loss仍然较高
   - Embedding质量不够好

## 预期效果

修复IDCG计算后：
- **NDCG应该显著提升**（可能提升2-5倍，取决于recall）
- **更接近论文基准值**（0.05-0.10范围）
- **反映真实的排序质量**

## 代码修复位置

所有修复都在 `src/train.py`：
- `evaluate_full` 函数（TF1.x兼容模式）
- `evaluate_full_keras` 函数（Keras模式）

总共修复了4处IDCG计算和NDCG累加逻辑。

## 下一步

1. **重新运行训练**，观察修复后的NDCG值
2. **对比修复前后的差异**
3. **如果NDCG仍然低于论文基准**：
   - 检查模型训练是否充分
   - 检查recall是否正常（应该接近论文值）
   - 可能需要调整超参数或训练更长时间

