# ComiRec-DR Recall 低的问题分析与解决方案

## 问题描述

训练 ComiRec-DR 模型时出现以下情况：
- **Recall**: 0.061821 (6.18%) - **严重偏低**
- **NDCG**: 0.045488 (4.55%)
- **触发早停**: 在 iter 284000 时停止
- **最优checkpoint**: ckpt-130

**预期结果**（根据原始论文）：
- Recall@50 应该在 **8.1%** 左右
- 当前只有 **6.2%**，相差约 25%

## 根本原因分析

### 问题 1: ⚠️ **学习率不正确** ⭐⭐⭐

**当前配置**:
```bash
--learning_rate 0.001
```

**正确配置**（来自 README.md）:
```
When training a ComiRec-DR model, you should set `--learning_rate 0.005`.
```

**影响**:
- ❌ lr=0.001 对 ComiRec-DR 来说**太小**
- ❌ 模型学习速度慢，难以收敛到最优解
- ❌ 容易陷入局部最优
- ❌ 导致 recall 偏低

**解决方案**: 使用 **lr=0.005** 重新训练

---

### 问题 2: 早停机制过早触发

**当前配置**:
```python
patience = 50  # 默认值
```

**现象**:
- 最优模型在 ckpt-130（约 130,000 次迭代）
- 之后 50 次评估（50,000 次迭代）内无提升
- 在 284,000 次迭代时触发早停

**分析**:
- Loss 从 7.34 仍在缓慢下降
- 模型可能还有提升空间，但 patience=50 太保守

**解决方案**: 
- 增加 `--patience 100` 或 `--patience 150`
- 或者使用更大的学习率（0.005）后重新训练

---

### 问题 3: 负样本数量偏少

**当前配置**:
```python
neg_num = 10
num_sampled = neg_num * batch_size = 10 * 128 = 1280
item_count = 367983
ratio = 1280 / 367983 = 0.35%
```

**影响**:
- 负样本只占总item的 0.35%，太少
- 模型难以学习区分相似item
- 可能导致 embedding 质量不高

**解决方案**: 增加负样本数量
```python
neg_num = 20  # 或者 30
```

---

## 解决方案

### 方案 1: 使用正确的学习率重新训练 ⭐ **强烈推荐**

**命令**:
```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --embedding_dim 64 \
    --hidden_size 64 \
    --num_interest 4 \
    --max_iter 1000 \
    --patience 100
```

**预期效果**:
- ✅ Recall 应该提升到 7-8% 以上
- ✅ 训练速度更快，更容易收敛
- ✅ 避免局部最优

**注意**: 这将从头开始训练，不会使用之前的 checkpoint

---

### 方案 2: 从现有 checkpoint 继续训练（需要修改代码）

当前代码已经支持自动恢复 checkpoint，但如果想要继续训练需要：

**步骤 1**: 确认 checkpoint 路径
```
best_model/book_ComiRec-DR_b128_lr0.001_d64_len20_test79/keras_ckpt/ckpt-130
```

**步骤 2**: 使用相同的实验名称重新运行
```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.001 \
    --max_iter 2000 \
    --patience 100
```

当提示输入实验名称时，输入 `test79`（与之前相同）

**步骤 3**: 程序会自动从 checkpoint 恢复并继续训练

**缺点**: 
- ⚠️ 仍然使用 lr=0.001（不推荐）
- ⚠️ 可能仍然停留在局部最优

---

### 方案 3: 微调学习率继续训练（推荐）

**步骤 1**: 修改代码以支持学习率调整

在 `src/train.py` 的训练函数中添加学习率调整功能：

```python
# 在恢复 checkpoint 后调整学习率
if latest_ckpt:
    print(f"Restoring from checkpoint: {latest_ckpt}")
    ckpt.restore(latest_ckpt)
    # 调整学习率（微调）
    optimizer.learning_rate.assign(lr * 0.5)  # 使用更小的学习率微调
    print(f"Adjusted learning rate to: {optimizer.learning_rate.numpy()}")
```

**步骤 2**: 使用稍高的学习率
```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.002 \
    --max_iter 2000 \
    --patience 100
```

使用实验名称 `test79`，程序会恢复并使用新的学习率继续训练。

---

### 方案 4: 增加负样本数量（配合方案1或3）

修改 `src/train.py` 第 286 行：
```python
neg_num = 20  # 从 10 增加到 20
```

或者添加命令行参数：
```python
parser.add_argument('--neg_num', type=int, default=10, help='negative sample number per positive')
```

然后：
```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --neg_num 20 \
    --max_iter 1000 \
    --patience 100
```

---

## 推荐行动方案

### 立即执行（最简单）⭐⭐⭐

**使用正确的学习率重新训练**:
```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --max_iter 1000 \
    --patience 100
```

输入新的实验名称（如 `test80`）开始训练。

**预期时间**: 
- 如果使用 GPU，大约 2-4 小时
- 应该在 200-300 个评估点内达到最优

**预期结果**:
- Recall@50: 7.5-8.5%
- NDCG@50: 12-14%

---

### 进阶优化（更好的结果）⭐⭐

1. **修改代码增加负样本数量**
2. **使用正确的学习率**
3. **增加 patience**

详见下面的代码修改部分。

---

## 代码修改（自动实现）

### 修改 1: 添加 neg_num 参数

在 `src/train.py` 的参数定义部分添加：
```python
parser.add_argument('--neg_num', type=int, default=10, help='negative samples per batch')
```

在 train 函数中使用：
```python
neg_num = args.neg_num  # 而不是硬编码为 10
```

### 修改 2: 支持学习率微调

在恢复 checkpoint 后调整学习率（如果需要）：
```python
if latest_ckpt:
    print(f"Restoring from checkpoint: {latest_ckpt}")
    ckpt.restore(latest_ckpt)
    # 可选：调整学习率用于微调
    if args.finetune_lr is not None:
        optimizer.learning_rate.assign(args.finetune_lr)
        print(f"Fine-tuning with learning rate: {args.finetune_lr}")
```

### 修改 3: 提高默认 patience（针对 ComiRec-DR）

在代码中添加自适应 patience：
```python
# 根据模型类型调整 patience
if model_type in ['ComiRec-DR', 'ComiRec-SA', 'MIND']:
    if patience == 50:  # 如果使用默认值
        patience = 100  # 多兴趣模型需要更多耐心
        print(f"Adjusted patience to {patience} for multi-interest model")
```

---

## 对比分析

### 当前结果 vs 预期结果

| 指标 | 当前结果 | 预期结果 | 差距 |
|------|---------|---------|------|
| Recall@50 | 6.18% | 8.10% | **-23.7%** |
| NDCG@50 | 4.55% | 13.52% | **-66.3%** |
| Hitrate | 13.32% | ~30%+ | **-55.6%** |

**结论**: 当前结果严重偏低，主要原因是学习率不正确。

---

## 训练监控建议

在训练时注意观察：

1. **Loss 曲线**:
   - 应该持续下降
   - 理想情况下降到 5.5-6.0 左右

2. **Recall 曲线**:
   - 前期应该快速上升
   - 在 100-200 个评估点内达到 6-7%
   - 最终稳定在 7.5-8.5%

3. **早停触发时机**:
   - 应该在 recall 达到 7.5%+ 后才触发
   - 如果在 6% 就停止，说明配置有问题

---

## 总结

**核心问题**: 使用了错误的学习率（0.001 而不是 0.005）

**解决方案**: 
1. ✅ 使用 `--learning_rate 0.005` 重新训练（最简单）
2. ⚠️ 或者增加 patience 和 neg_num，从 checkpoint 继续训练（次优）

**预期改善**:
- Recall: 6.18% → **8.0%+** (提升 30%)
- NDCG: 4.55% → **13%+** (提升 3倍)
- 训练时间: 更快收敛

**立即执行**:
```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --patience 100 \
    --max_iter 1000
```

