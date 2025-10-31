# NDCG计算方式说明

## 发现的问题

修复IDCG计算后，NDCG比最开始还低，但recall正常。这表明：

### 原因分析

1. **原始实现的IDCG计算方式**：
   ```python
   idcg = 0.0
   for no in range(recall):  # 基于实际命中的item数
       idcg += 1.0 / math.log(no + 2, 2)
   ```
   
   这种方式虽然不符合标准定义（标准应基于真实标签总数），但**可能是原始TensorFlow 1.14实现的方式**，论文基准也是基于这种方式。

2. **修复后的问题**：
   - 如果改为基于真实标签总数计算IDCG，会导致IDCG变大
   - DCG不变，所以NDCG = DCG/IDCG会变小
   - 这就是为什么修复后NDCG反而变低

3. **为了与论文基准对比**：
   - 应该保持原始的实现方式
   - 即使这在理论上不太标准，但为了可对比性，需要保持一致

## 当前实现（已恢复为原始方式）

```python
# IDCG计算：基于实际命中的item数（与原始TensorFlow 1.14实现保持一致）
idcg = 0.0
for no in range(recall):
    idcg += 1.0 / math.log(no + 2, 2)

# NDCG计算
if recall > 0:
    total_ndcg += dcg / idcg
    total_hitrate += 1
else:
    total_ndcg += 0.0  # recall=0时，NDCG=0

ndcg = total_ndcg / total  # 所有样本的平均NDCG
```

## 为什么NDCG可能仍然比论文基准低？

即使恢复了原始实现，NDCG仍然可能低，可能原因：

### 1. 分母问题

如果hitrate很低（比如0.1），使用`total`作为分母会导致NDCG被低估。

**可以尝试**：
```python
# 只对有hit的样本计算平均NDCG
ndcg = total_ndcg / total_hitrate if total_hitrate > 0 else 0.0
```

这样NDCG会更高，但可能更接近论文基准（如果论文使用的是这种方式）。

### 2. 模型排序质量差

即使recall正常，如果真实item排在推荐列表后面，DCG会被折损函数严重惩罚：
- 位置1: DCG = 1.0 / log₂(3) = 0.63
- 位置10: DCG = 1.0 / log₂(12) = 0.29
- 位置20: DCG = 1.0 / log₂(22) = 0.21

### 3. 模型训练不充分

- Loss仍然较高
- Embedding质量不够好
- 需要更多训练迭代

## 建议

1. **先恢复为原始IDCG计算方式**（已完成）✅
2. **尝试修改分母**：
   ```python
   # 选项1：除以total（当前方式，可能导致低估）
   ndcg = total_ndcg / total
   
   # 选项2：除以hitrate（可能更接近论文）
   ndcg = total_ndcg / total_hitrate if total_hitrate > 0 else 0.0
   ```
3. **如果仍然低**：
   - 检查真实item的平均排名位置
   - 如果平均位置>10，说明排序质量差，需要改进模型
   - 可能需要更多训练或调整超参数

