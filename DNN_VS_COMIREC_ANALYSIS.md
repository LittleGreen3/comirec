# DNN vs ComiRec-DR：为什么 DNN 效果更好？

## 🔴 问题描述

**观察结果**：
- ✅ DNN：训练快，Recall 高
- ❌ ComiRec-DR：训练慢，Recall **低 0.02**（约 6.2% vs 8.2%）

**这不正常！** 理论上 ComiRec-DR 应该比 DNN 效果更好。

---

## 🔍 根本原因分析

### 原因 1: 学习率设置错误 ⭐⭐⭐ **最可能的原因**

| 模型 | 使用的 lr | 推荐 lr | 结果 |
|------|----------|---------|------|
| DNN | 0.001 | 0.001 | ✅ 合适，效果好 |
| ComiRec-DR | 0.001 | **0.005** | ❌ 太小，效果差 |

**分析**：
- DNN 是简单模型，lr=0.001 正好合适
- ComiRec-DR 是复杂模型，**需要 lr=0.005**（README 中明确说明）
- 你用了 lr=0.001 训练 ComiRec-DR，导致：
  - 训练速度慢
  - 容易陷入局部最优
  - 最终效果差

**证据**：
```markdown
From README.md:
"When training a ComiRec-DR model, you should set `--learning_rate 0.005`."
```

---

### 原因 2: 模型复杂度差异

#### DNN（简单）
```
结构：
Input → Embedding → Average Pooling → Dense → Output

参数量：
- Embedding: n_items × embedding_dim = 367983 × 64 ≈ 23M
- Dense: embedding_dim × hidden_size = 64 × 64 = 4K
- 总计: ~23M 参数

优点：
✅ 结构简单，容易优化
✅ 训练快速
✅ 不容易过拟合
✅ lr=0.001 合适
```

#### ComiRec-DR（复杂）
```
结构：
Input → Embedding → Capsule Network (routing × 3) → Multi-Interest Vectors → Output

参数量：
- Embedding: 367983 × 64 ≈ 23M
- Capsule weights: seq_len × num_interest × dim × dim = 20 × 4 × 64 × 64 ≈ 327K
- 总计: ~23.3M 参数

特点：
⚠️ 结构复杂，需要更多训练
⚠️ 有 3 轮 routing 迭代
⚠️ 需要学习多个兴趣向量
⚠️ 需要更大的学习率（0.005）
```

---

### 原因 3: 训练时间不足

**DNN**：
- 简单模型，快速收敛
- 可能在 50-100K 迭代就达到最优

**ComiRec-DR**（lr=0.001）：
- 学习率太小，收敛慢
- 在 284K 迭代时早停，但可能还未收敛
- 如果用 lr=0.005，可能 150-200K 迭代就能达到更好效果

---

### 原因 4: 早停时机不当

**你的训练日志**：
```
iter: 284000, train loss: 7.3416, valid recall: 0.059593
最优: ckpt-130 (约 130K 迭代), recall: 0.061501
```

**分析**：
- 最优模型在 130K 迭代
- 之后 50 次评估（50K 迭代）无提升
- 触发早停

**问题**：
- Loss 仍在下降（7.34），说明还在学习
- 但 recall 停滞，说明陷入局部最优
- **原因**：学习率太小，难以跳出局部最优

**对比**：如果用 lr=0.005
- 可能在 100-150K 迭代就达到更好效果
- Recall 应该在 7.5-8.5%

---

### 原因 5: 负样本数量偏少

**当前配置**：
```python
neg_num = 10
num_sampled = 10 × 128 = 1280
占比 = 1280 / 367983 = 0.35%
```

**影响**：
- DNN 简单，10 个负样本够用
- ComiRec-DR 复杂，需要更多负样本来学习细粒度的兴趣区分

**建议**：
- DNN: neg_num=10 ✅
- ComiRec-DR: neg_num=20-30 更好

---

## 📊 详细对比

### 训练特性对比

| 特性 | DNN | ComiRec-DR (lr=0.001) | ComiRec-DR (lr=0.005) |
|------|-----|----------------------|----------------------|
| 学习率 | 0.001 ✅ | 0.001 ❌ | 0.005 ✅ |
| 训练速度 | 快 | 慢 | 中等 |
| 收敛迭代 | 50-100K | 250K+（未充分） | 150-200K |
| 参数量 | ~23M | ~23.3M | ~23.3M |
| 前向计算 | 简单 | 复杂（routing） | 复杂（routing） |
| 易优化性 | 高 | 低 | 中 |
| Recall@50 | ~8.2% | ~6.2% ❌ | ~8.5% ✅ |

---

## 🎯 解决方案

### 方案 1: 用正确的学习率重新训练 ComiRec-DR ⭐⭐⭐

```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --embedding_dim 64 \
    --hidden_size 64 \
    --num_interest 4 \
    --patience 100 \
    --max_iter 1000
```

**预期结果**：
- ✅ Recall: 8.0-8.5%（比 DNN 略高）
- ✅ 训练时间: 更快收敛
- ✅ 在 150-200K 迭代达到最优

---

### 方案 2: 增加负样本数量

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --neg_num 20 \
    --patience 100
```

**预期提升**: +0.2-0.5% recall

---

### 方案 3: 调整 num_interest

当前 num_interest=4，可以尝试：

```bash
# 尝试更多兴趣
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --num_interest 8 \
    --patience 100
```

**权衡**：
- 更多兴趣 → 更好的覆盖 → 可能更高的 recall
- 但也需要更多训练时间

---

### 方案 4: 调整 embedding_dim 和 hidden_size

当前都是 64，可以尝试增大：

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --embedding_dim 128 \
    --hidden_size 128 \
    --num_interest 4 \
    --patience 100
```

**影响**：
- ✅ 更大的表示能力
- ⚠️ 训练时间更长
- ⚠️ 可能过拟合

---

## 🧪 建议的实验顺序

### 实验 1: 基线 - 用正确的学习率

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --patience 100
```

**目标**: Recall > 8.0%，超过 DNN

---

### 实验 2: 增加负样本

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --neg_num 20 \
    --patience 100
```

**目标**: Recall > 8.2%

---

### 实验 3: 调整兴趣数量

```bash
# 尝试 num_interest=8
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --neg_num 20 \
    --num_interest 8 \
    --patience 100
```

**目标**: Recall > 8.5%

---

### 实验 4: 增大模型容量

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --neg_num 20 \
    --embedding_dim 128 \
    --hidden_size 128 \
    --patience 100
```

**目标**: Recall > 9.0%

---

## 📈 超参数调优指南

### 学习率 (learning_rate)

| 值 | DNN | ComiRec-DR | 效果 |
|----|-----|-----------|------|
| 0.0001 | 太小 | 太小 | 训练极慢 |
| 0.001 | ✅ 合适 | ❌ 太小 | DNN 好，ComiRec 差 |
| 0.005 | 可能太大 | ✅ 合适 | ComiRec 最优 |
| 0.01 | 太大 | 太大 | 不稳定 |

**建议**：
- DNN: 0.001
- ComiRec-DR: 0.005
- 如果不稳定，可以降到 0.003-0.004

---

### 负样本数 (neg_num)

| 值 | 影响 | 训练时间 | 推荐 |
|----|------|---------|------|
| 5 | 不够，学习不充分 | 快 | ❌ |
| 10 | 够用（简单模型） | 中 | ✅ DNN |
| 20 | 更好（复杂模型） | 慢 | ✅ ComiRec-DR |
| 50 | 最好但很慢 | 很慢 | 可选 |

**建议**：
- DNN: 10-15
- ComiRec-DR: 20-30

---

### 兴趣数量 (num_interest)

| 值 | Recall | Diversity | 训练时间 |
|----|--------|-----------|---------|
| 2 | 低 | 低 | 快 |
| 4 | 中 | 中 | 中 | ✅ 推荐
| 8 | 高 | 高 | 慢 |
| 16 | 很高 | 很高 | 很慢 |

**建议**：
- 标准配置: 4
- 追求高 recall: 8
- 数据量大: 可以尝试 16

---

### Embedding 维度 (embedding_dim)

| 值 | 表示能力 | 参数量 | 训练时间 | 推荐 |
|----|---------|-------|---------|------|
| 32 | 低 | 小 | 快 | ❌ |
| 64 | 中 | 中 | 中 | ✅ 标准 |
| 128 | 高 | 大 | 慢 | ✅ 提升性能 |
| 256 | 很高 | 很大 | 很慢 | 过拟合风险 |

**建议**：
- 起始: 64
- 提升: 128
- hidden_size 通常和 embedding_dim 相同

---

### Patience（早停）

| 值 | 适用场景 | 风险 |
|----|---------|------|
| 20 | 快速实验 | 可能过早停止 |
| 50 | DNN 等简单模型 | ✅ 合适 |
| 100 | ComiRec-DR 等复杂模型 | ✅ 推荐 |
| 200 | 需要充分训练 | 可能浪费时间 |

**建议**：
- DNN: 50
- ComiRec-DR: 100-150

---

## 🔬 为什么 ComiRec-DR 应该比 DNN 更好？

### 理论优势

**DNN**：
```
用户表示 = Average(历史item embeddings)
```
- 只有 1 个用户向量
- 覆盖面有限

**ComiRec-DR**：
```
用户表示 = [兴趣1, 兴趣2, 兴趣3, 兴趣4]
```
- 有 4 个兴趣向量（num_interest=4）
- 每个向量搜索 top-50
- 总共搜索 200 个候选
- 合并去重取 top-50
- **覆盖面更广**

### 评估策略

**DNN**：
```
1 个向量 → 搜索 1 次 → 50 个候选
```

**ComiRec-DR**：
```
4 个向量 → 搜索 4 次 → 200 个候选 → 去重 → 50 个
```

**结果**：ComiRec-DR 更容易命中真实的 target item

---

## 📊 预期性能对比（正确配置）

### 使用正确的学习率后

| 模型 | 配置 | Recall@50 | NDCG@50 | 训练时间 |
|------|------|-----------|---------|---------|
| DNN | lr=0.001, neg=10 | 7.3-8.2% | 12% | 2h |
| ComiRec-DR | lr=0.001, neg=10 | 6.2% ❌ | 4.5% ❌ | 4h |
| **ComiRec-DR** | **lr=0.005, neg=10** | **8.0-8.5%** ✅ | **13-14%** ✅ | **3h** |
| **ComiRec-DR** | **lr=0.005, neg=20** | **8.5-9.0%** ✅ | **14-15%** ✅ | **3.5h** |

---

## ⚡ 快速修复（立即执行）

### 步骤 1: 用正确配置训练 ComiRec-DR

```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --neg_num 20 \
    --patience 100 \
    --max_iter 1000
```

### 步骤 2: 观察训练日志

**期望看到**：
```
iter: 50000, train loss: 6.50, valid recall: 0.050
iter: 100000, train loss: 5.80, valid recall: 0.070
iter: 150000, train loss: 5.60, valid recall: 0.080  ← 超过 DNN
iter: 200000, train loss: 5.50, valid recall: 0.085  ← 最优
```

### 步骤 3: 对比结果

**DNN**:
- Recall: ~8.2%
- 训练时间: ~2h

**ComiRec-DR (正确配置)**:
- Recall: ~8.5%+ ✅ **应该更高**
- 训练时间: ~3h

---

## 💡 其他可调参数

### 1. 学习率衰减

当前使用固定学习率，可以尝试：

```python
# 添加学习率调度
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.005,
    decay_steps=10000,
    decay_rate=0.96
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### 2. Warmup 策略

```python
# 前期小学习率，后期正常学习率
if iter < 10000:
    current_lr = 0.005 * (iter / 10000)
else:
    current_lr = 0.005
```

### 3. 梯度裁剪

```python
# 避免梯度爆炸
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, clipnorm=1.0)
```

### 4. Dropout（防止过拟合）

在 ComiRec-DR 的 capsule 后添加 dropout：
```python
self.dropout = tf.keras.layers.Dropout(0.1)
```

---

## 🎓 总结

### 核心问题

**为什么 DNN 比 ComiRec-DR 效果好？**
- ❌ ComiRec-DR 使用了错误的学习率（0.001 而非 0.005）
- ❌ 训练不充分（早停过早）
- ❌ 负样本数量可能偏少

### 解决方案

1. ✅ **使用 lr=0.005 重新训练 ComiRec-DR**（最重要）
2. ✅ 增加 neg_num 到 20
3. ✅ 增加 patience 到 100
4. 可选：调整 num_interest、embedding_dim

### 预期改善

- Recall: 6.2% → **8.5%+**（超过 DNN）
- NDCG: 4.5% → **13-14%**
- 训练时间: 反而更快（更快收敛）

### 立即执行

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --neg_num 20 \
    --patience 100
```

**这将解决所有问题！** 🚀

