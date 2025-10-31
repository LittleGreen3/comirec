# 快速修复：ComiRec-DR Recall 低的问题

## 🔴 问题：Recall 只有 6.2%，预期应该是 8.1%

## ⚡ 快速解决方案

### 方案 1: 使用正确的学习率重新训练 ⭐ **最简单，强烈推荐**

```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --patience 100 \
    --max_iter 1000
```

**为什么**: 
- ❌ 你之前使用的是 `lr=0.001`（错误）
- ✅ ComiRec-DR 官方推荐使用 `lr=0.005`（正确）
- 📈 预期 Recall 提升: 6.2% → **8.0%+**

**注意**: 输入新的实验名称（如 `test80`），从头开始训练。

---

### 方案 2: 从现有模型继续训练（但仍用错误的学习率）

```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.001 \
    --patience 150 \
    --max_iter 2000
```

输入相同的实验名称 `test79`，程序会自动从 checkpoint 继续。

⚠️ **缺点**: 仍然使用错误的学习率，效果有限。

---

## 📊 代码已自动改进

### 改进 1: 自动警告学习率问题

现在当你使用 `lr=0.001` 训练 ComiRec-DR 时，会显示：

```
================================================================================
⚠️  警告：ComiRec-DR 推荐使用 learning_rate=0.005
   当前使用 lr=0.001 可能导致：
   - 训练速度慢
   - 容易陷入局部最优
   - Recall 显著低于预期
   
   建议：使用 --learning_rate 0.005 重新训练
================================================================================
```

### 改进 2: 多兴趣模型自动调整 patience

对于 ComiRec-DR/SA/MIND 模型，默认 patience 自动从 50 → 100

```
⚠️  多兴趣模型自动调整 patience: 50 → 100
   原因：多兴趣模型需要更多训练时间来收敛
```

### 改进 3: 支持调整负样本数

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --neg_num 20 \
    --patience 100
```

### 改进 4: 更好的 checkpoint 恢复提示

```
✅ 发现已有 checkpoint，自动恢复: best_model/.../ckpt-130
   将从上次训练继续...
   当前学习率: 0.005
   当前 patience: 100
   负样本数: 10 (每个正样本)
```

---

## 🎯 推荐配置

### 标准配置（推荐）

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

### 优化配置（更好的结果）

```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --embedding_dim 64 \
    --hidden_size 64 \
    --num_interest 4 \
    --neg_num 20 \
    --patience 100 \
    --max_iter 1000
```

---

## 📈 预期结果

### 使用 lr=0.001（错误）
- Recall@50: 6.18%
- NDCG@50: 4.55%
- 训练很慢，容易停滞

### 使用 lr=0.005（正确）
- Recall@50: **8.0-8.5%** ✅
- NDCG@50: **12-14%** ✅
- 训练更快，收敛更好

---

## 💡 为什么会出现这个问题？

1. **README.md 中有说明**，但容易被忽略：
   ```
   When training a ComiRec-DR model, you should set `--learning_rate 0.005`.
   ```

2. **默认学习率是 0.001**，对 DNN/GRU4REC 合适，但对 ComiRec-DR 太小

3. **学习率太小导致**：
   - 模型更新步长太小
   - 难以跳出局部最优
   - 需要更多迭代才能收敛
   - 最终 recall 显著偏低

---

## 🔍 如何验证修复

训练时观察：

### ✅ 正常情况（lr=0.005）
```
iter: 50000, train loss: 6.50, valid recall: 0.045
iter: 100000, train loss: 5.80, valid recall: 0.065
iter: 150000, train loss: 5.60, valid recall: 0.075
iter: 200000, train loss: 5.50, valid recall: 0.080  ← 达到预期
```

### ❌ 异常情况（lr=0.001）
```
iter: 50000, train loss: 7.20, valid recall: 0.025
iter: 100000, train loss: 6.80, valid recall: 0.045
iter: 150000, train loss: 6.60, valid recall: 0.055
iter: 200000, train loss: 6.50, valid recall: 0.060  ← 偏低，停滞
```

---

## ❓ 常见问题

### Q: 能从之前的模型继续训练吗？

**A**: 可以，但不推荐。之前的模型用错误的学习率训练，可能已陷入局部最优。
建议用正确的学习率重新训练。

### Q: 训练需要多久？

**A**: 
- GPU: 2-4 小时
- 预计在 200-300 个评估点（200K-300K 次迭代）内达到最优

### Q: 为什么原来的训练在 284000 次迭代时停止？

**A**: 
- 最优模型在 ~130K 迭代（ckpt-130）
- 之后 50 次评估（50K 迭代）内无提升
- 触发早停机制

用正确的学习率，应该在更早的迭代数就能达到更好的结果。

### Q: 需要修改其他参数吗？

**A**: 基本不需要，但可以尝试：
- `--neg_num 20` 增加负样本（可能略微提升）
- `--num_interest 8` 增加兴趣数量（需要更多训练时间）

---

## 📝 总结

**核心问题**: 学习率设置错误（0.001 vs 0.005）

**解决方案**: 
```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --patience 100
```

**预期改善**: Recall 从 6.2% → 8.0%+（提升 30%）

**立即执行**: 复制上面的命令，开始新的训练！

---

详细分析请查看：`COMIREC_DR_LOW_RECALL_FIX.md`

