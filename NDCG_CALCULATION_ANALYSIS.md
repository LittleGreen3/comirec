# NDCG 计算分析和问题诊断

## NDCG 计算公式

### 标准公式

```
NDCG = DCG / IDCG
```

其中：
- **DCG (Discounted Cumulative Gain)**: 折扣累积增益
  ```
  DCG = Σ (rel_i / log₂(i + 2))
  ```
  其中 `i` 是推荐位置（从0开始），`rel_i` 是第i个位置的关联度（这里命中为1，未命中为0）

- **IDCG (Ideal DCG)**: 理想情况下的DCG（所有相关item都在最前面）
  ```
  IDCG = Σ (1 / log₂(i + 2)) for i in [0, 1, 2, ..., num_relevant_items - 1]
  ```

## 当前代码的NDCG计算

### 代码逻辑（第103-220行）

```python
# 对每个样本
if recall > 0:
    total_ndcg += dcg / idcg  # 只对recall>0的样本累加NDCG
    total_hitrate += 1

# 最终计算
ndcg = total_ndcg / total  # ⚠️ 问题在这里！
```

### 问题分析

**当前实现的问题**：

1. **分母使用总样本数**：`ndcg = total_ndcg / total`
   - `total_ndcg` 只包含 `recall > 0` 的样本的NDCG之和
   - `total` 是**所有**样本数（包括 `recall = 0` 的样本）

2. **导致NDCG被低估**：
   ```
   假设：
   - 总样本数 total = 1000
   - recall > 0 的样本数 = 100（hitrate = 0.1）
   - 这100个样本的平均NDCG = 0.5
   
   当前计算：
   ndcg = (100 × 0.5) / 1000 = 0.05  ❌ 被低估！
   
   正确计算应该是：
   ndcg = (100 × 0.5) / 100 = 0.5  ✅ 这才是真实的NDCG
   ```

3. **为什么会这样**：
   - 原始代码可能是为了处理recall=0的情况（NDCG=0/0=NaN）
   - 但这种方式会导致NDCG被严重低估
   - 特别是当hitrate很低时（如0.1），NDCG会被低估10倍！

## NDCG低的主要原因

### 1. 计算方式导致被低估（最严重）

如上所述，分母应该是 `total_hitrate` 而不是 `total`。

### 2. Recall低导致IDCG小

如果recall很低（比如只命中1-2个item），IDCG也会很小：
```
只命中1个item: IDCG = 1.0 / log₂(2) = 1.0
只命中2个item: IDCG = 1.0 / log₂(2) + 1.0 / log₂(3) = 1.0 + 0.63 = 1.63
命中5个item:  IDCG ≈ 2.79
```

### 3. 真实item排在后面

即使命中了真实item，但如果排在推荐列表后面，DCG会被折损函数惩罚：
```
位置1:  DCG贡献 = 1.0 / log₂(3) = 0.63
位置10: DCG贡献 = 1.0 / log₂(12) = 0.29
位置50: DCG贡献 = 1.0 / log₂(52) = 0.19
```

### 4. 模型排序质量差

如果模型学到的embedding质量不好，Faiss搜索返回的排序可能不准确，真实item可能排在后面。

## 修复建议

### 方案1：使用hitrate作为分母（推荐）

```python
# 修复前
ndcg = total_ndcg / total  # 错误：被低估

# 修复后
ndcg = total_ndcg / total_hitrate if total_hitrate > 0 else 0.0  # 正确
```

**优点**：
- NDCG反映的是"有hit的样本的平均NDCG"
- 与hitrate指标一致

**缺点**：
- 忽略了recall=0的样本

### 方案2：所有样本都计入NDCG（最正确）

```python
# 对每个样本
if recall > 0:
    total_ndcg += dcg / idcg
else:
    total_ndcg += 0.0  # recall=0时，NDCG=0

# 最终计算
ndcg = total_ndcg / total  # 现在正确了
```

**优点**：
- 最符合NDCG的定义
- 所有样本都计入统计

**缺点**：
- NDCG会更低（因为recall=0的样本NDCG=0）

### 方案3：分别报告两个指标

```python
ndcg_hit = total_ndcg / total_hitrate if total_hitrate > 0 else 0.0  # 有hit样本的平均NDCG
ndcg_all = total_ndcg / total  # 所有样本的平均NDCG（recall=0时NDCG=0）
```

## 与原始论文的对比

根据原始论文的基准结果（TensorFlow 1.14）：
- **Amazon Books, ComiRec-DR**: NDCG@20 = 9.185%（即0.09185）
- **Amazon Books, DNN**: NDCG@20 = 7.670%（即0.07670）

如果你的NDCG远低于这个值（比如0.01-0.05），很可能是计算方式的问题。

## 诊断步骤

1. **检查hitrate**：
   ```python
   hitrate = total_hitrate / total
   ```
   如果hitrate很低（<0.1），说明recall很低，这是根本问题。

2. **检查NDCG计算**：
   ```python
   # 当前（可能有误）
   ndcg_current = total_ndcg / total
   
   # 应该（正确）
   ndcg_correct = total_ndcg / total_hitrate if total_hitrate > 0 else 0.0
   
   # 对比
   print(f"当前NDCG: {ndcg_current:.6f}")
   print(f"正确NDCG: {ndcg_correct:.6f}")
   print(f"差异倍数: {ndcg_correct / ndcg_current:.2f}x" if ndcg_current > 0 else "N/A")
   ```

3. **检查排序质量**：
   - 查看真实item在推荐列表中的平均位置
   - 如果平均位置>10，说明排序质量差

## 预期修复后的效果

修复后，NDCG应该：
- **显著提升**（如果之前被低估了10倍，修复后会提升10倍）
- **更接近论文基准值**（如0.05-0.10范围）
- **反映真实的排序质量**

## 代码修复位置

- `src/train.py` 第214行：`ndcg = total_ndcg / total`
- `src/train.py` 第320行：`ndcg = total_ndcg / total`（Keras版本）

需要修改为：
```python
ndcg = total_ndcg / total_hitrate if total_hitrate > 0 else 0.0
```

或者使用方案2，对所有样本计算NDCG。

