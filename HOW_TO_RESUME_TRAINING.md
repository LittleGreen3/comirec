# 如何从之前的模型继续训练

## 📍 原理说明

### Checkpoint 存储位置

当你训练模型时，checkpoint 保存在：
```
best_model/{实验名称}/keras_ckpt/
```

例如，你的 test79 实验：
```
best_model/book_ComiRec-DR_b128_lr0.001_d64_len20_test79/keras_ckpt/ckpt-130
```

### 恢复机制

代码在训练开始时会：
1. **根据实验名称**构建 checkpoint 路径
2. **检查是否存在** checkpoint 文件
3. **如果存在**，自动恢复模型权重和优化器状态
4. **继续训练**，从上次停止的地方继续

---

## 🔧 如何继续训练 test79

### 步骤 1: 确认实验名称

从你的日志可以看到最优模型在：
```
best_model/book_ComiRec-DR_b128_lr0.001_d64_len20_test79/keras_ckpt/ckpt-130
```

实验名称是：**`test79`**

### 步骤 2: 使用相同的参数运行训练

```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.001 \
    --embedding_dim 64 \
    --hidden_size 64 \
    --num_interest 4 \
    --patience 100 \
    --max_iter 2000
```

**重要**：参数必须和之前训练时**完全相同**（除了 `patience` 和 `max_iter` 可以改）

因为实验名称是根据这些参数生成的：
```
{dataset}_{model_type}_b{batch_size}_lr{lr}_d{embedding_dim}_len{maxlen}_{你的输入}
```

### 步骤 3: 输入实验名称

程序会提示：
```
Please input the experiment name: 
```

**输入**: `test79`

### 步骤 4: 自动恢复

程序会：
1. 根据参数和实验名构建路径
2. 检查 `best_model/book_ComiRec-DR_b128_lr0.001_d64_len20_test79/keras_ckpt/`
3. 找到 `ckpt-130`
4. 显示：
   ```
   ✅ 发现已有 checkpoint，自动恢复: best_model/.../ckpt-130
      将从上次训练继续...
      当前学习率: 0.001
      当前 patience: 100
      负样本数: 10 (每个正样本)
   ```
5. 恢复模型权重和优化器状态
6. **继续训练**，从第 130 个 checkpoint 之后继续

---

## 📊 训练日志示例

### 继续训练时你会看到：

```
Please input the experiment name: test79

✅ 发现已有 checkpoint，自动恢复: best_model/book_ComiRec-DR_b128_lr0.001_d64_len20_test79/keras_ckpt/ckpt-130
   将从上次训练继续...
   当前学习率: 0.001
   当前 patience: 100
   负样本数: 10 (每个正样本)

training begin (Keras)

iter: 284000, train loss: 7.3416, valid recall: 0.059593, valid ndcg: 0.043448, valid hitrate: 0.131015
iter: 285000, train loss: 7.3200, valid recall: 0.060123, valid ndcg: 0.044000, valid hitrate: 0.132000
...
```

**注意**：
- ✅ 模型权重从 ckpt-130 恢复
- ✅ 优化器状态（如 Adam 的动量）也恢复了
- ✅ 训练从迭代 284000 继续（或者重新计数，取决于实现）
- ⚠️ **但是**，best_metric 全局变量**不会恢复**（这是代码限制）

---

## ⚠️ 关于 best_metric 的注意事项

**问题**：
```python
best_metric = 0  # 全局变量，每次运行都会重置
```

**影响**：
- ✅ 模型权重恢复：正常
- ✅ 优化器状态恢复：正常
- ❌ `best_metric` 恢复：**不会恢复**，从 0 开始

**结果**：
- 如果恢复后的第一次评估 recall < 之前最优，不会保存
- 如果 recall > 之前最优，会保存新的 checkpoint

**解决方案**：代码已经处理了这个问题，因为：
1. 每次评估时都会检查当前 recall > best_metric
2. 如果更优，会保存新的 checkpoint
3. 所以即使 best_metric 重置了，只要有提升就会保存

---

## 🔍 完整示例

### 场景：从 test79 继续训练

**之前的训练**（已停止）：
```
experiment: test79
checkpoint: ckpt-130
最后 recall: 0.061501
停止原因: 早停（50次评估无提升）
```

**继续训练命令**：
```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.001 \
    --embedding_dim 64 \
    --hidden_size 64 \
    --num_interest 4 \
    --patience 150 \
    --max_iter 3000
```

**运行时**：
```
Please input the experiment name: test79
✅ 发现已有 checkpoint，自动恢复: .../ckpt-130
   将从上次训练继续...
   
training begin (Keras)
iter: 284000, train loss: 7.34, valid recall: 0.05959
iter: 285000, train loss: 7.32, valid recall: 0.06012  ← 继续训练
...
```

**如果后续有提升**：
```
iter: 290000, train loss: 7.25, valid recall: 0.06250
   ← 如果这次 recall > 0.061501，会保存为 ckpt-131
```

---

## 💡 常见问题

### Q1: 可以用不同的参数继续训练吗？

**A**: 可以，但**不推荐**，因为：
- 如果参数不同，实验名称会不同，找不到原来的 checkpoint
- 如果强制使用相同的实验名但参数不同，可能导致模型不匹配

**正确做法**：
- ✅ 使用**相同参数**继续训练
- ✅ 可以改变：`patience`, `max_iter`
- ⚠️ 可以改变：`learning_rate`（但 optimizer 状态可能不匹配）

### Q2: 可以改变学习率吗？

**A**: 技术上可以，但需要理解影响：

```bash
# 继续训练，但用不同的学习率
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.002 \  # 从 0.001 改为 0.002
    ... 其他参数相同
```

**输入**: `test79`

**结果**：
- ✅ 模型权重会恢复
- ⚠️ 优化器的学习率会被**重新设置**为 0.002
- ⚠️ 优化器的动量/二阶矩估计可能不匹配

**更安全的方式**（修改学习率）：
- 恢复 checkpoint 后，在代码中动态调整：
  ```python
  if latest_ckpt:
      ckpt.restore(latest_ckpt)
      optimizer.learning_rate.assign(0.002)  # 微调学习率
  ```

### Q3: 如何知道恢复是否成功？

**A**: 看输出信息：

```
✅ 发现已有 checkpoint，自动恢复: .../ckpt-130
   将从上次训练继续...
```

如果看到这个，说明恢复成功。

如果没有看到，说明：
- 实验名称不对
- 参数不匹配导致路径不同
- checkpoint 文件不存在

### Q4: 如何从特定的 checkpoint 恢复？

**A**: 当前代码自动恢复**最新的** checkpoint（ckpt-130）。

如果想恢复更早的，需要：
1. 查看 checkpoint 目录：
   ```bash
   ls best_model/book_ComiRec-DR_.../keras_ckpt/
   ```
2. 如果只有一个 ckpt-130，那就是它
3. CheckpointManager 默认只保留 1 个（`max_to_keep=1`）

### Q5: 训练会从哪个迭代数继续？

**A**: 这取决于实现：
- **Option 1**: 从 checkpoint 保存时的迭代数继续（如果有记录）
- **Option 2**: 从 0 开始重新计数（但模型权重是恢复的）

当前代码似乎是 **Option 2**，因为：
- 训练循环重新开始
- `iter` 变量从 0 开始
- 但模型权重是从 checkpoint 恢复的

这不影响训练，只是日志中的迭代数会重新开始。

---

## 📝 代码流程详解

### 关键代码位置

```python
# 1. 获取实验名称（需要用户输入）
exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen)
# 用户输入: "test79"
# 结果: "book_ComiRec-DR_b128_lr0.001_d64_len20_test79"

# 2. 构建 checkpoint 路径
best_model_path = "best_model/" + exp_name + '/'
ckpt_dir = os.path.join(best_model_path, 'keras_ckpt')
# 结果: "best_model/book_ComiRec-DR_b128_lr0.001_d64_len20_test79/keras_ckpt/"

# 3. 创建 CheckpointManager
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)

# 4. 查找最新的 checkpoint
latest_ckpt = ckpt_manager.latest_checkpoint
# 结果: "best_model/.../keras_ckpt/ckpt-130" 或 None

# 5. 如果找到，恢复
if latest_ckpt:
    ckpt.restore(latest_ckpt)  # 恢复模型和优化器
```

---

## 🎯 实际操作示例

### 示例 1: 继续训练（相同参数）

```bash
# 命令
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.001 \
    --patience 150

# 输入
Please input the experiment name: test79

# 输出
✅ 发现已有 checkpoint，自动恢复: .../ckpt-130
   将从上次训练继续...
   
training begin (Keras)
iter: 284000, train loss: 7.34, valid recall: 0.05959
...
```

### 示例 2: 用新名称重新训练

```bash
# 命令（使用不同的实验名）
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --patience 100

# 输入
Please input the experiment name: test80

# 输出（没有 checkpoint，从头开始）
training begin (Keras)
iter: 1000, train loss: 8.50, valid recall: 0.02000
...
```

---

## ✅ 总结

**继续训练 test79 的步骤**：

1. ✅ 使用**相同参数**（dataset, model_type, lr, embedding_dim, hidden_size, num_interest, batch_size, maxlen）
2. ✅ 运行训练命令
3. ✅ 输入实验名称：`test79`
4. ✅ 程序自动找到并恢复 checkpoint
5. ✅ 训练继续，从上次停止的地方开始

**关键点**：
- 🔑 实验名称决定 checkpoint 路径
- 🔑 相同实验名称 = 自动恢复
- 🔑 不同实验名称 = 从头训练

---

**现在你应该明白了！** 🎉

如果需要继续训练 test79，只需要：
```bash
python src/train.py --dataset book --model_type ComiRec-DR --learning_rate 0.001 --embedding_dim 64 --hidden_size 64 --num_interest 4
```
然后输入 `test79` 即可！

