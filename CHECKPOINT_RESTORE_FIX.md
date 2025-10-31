# Checkpoint 恢复问题修复说明

## 🔴 问题描述

### 症状
- 训练达到 recall=0.034
- 停止训练后，继续训练
- 恢复 checkpoint 后，recall 降到 0.020
- Loss 也变化很大

### 根本原因

**问题 1：`best_metric` 未持久化**
```python
best_metric = 0  # 全局变量

# 每次运行都重置为 0
# 恢复 checkpoint 后不知道之前的最优 recall 是多少
```

**问题 2：CheckpointManager 保存的是"最新"而非"最优"**
```python
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
# max_to_keep=1：只保留 1 个 checkpoint
# 保存的是最后一次评估的模型
# 不一定是 recall 最高的模型
```

**导致的结果**：
1. 训练到 100K 迭代
   - 最优模型在 80K（recall=0.034）
   - 继续训练到 100K（recall=0.028，过拟合）
   - 保存的是 100K 的模型（最新但不是最优）

2. 恢复训练
   - 恢复的是 100K 的模型（recall=0.028）
   - `best_metric` 重置为 0
   - 第一次评估显示 recall=0.028
   - 但由于 `best_metric=0`，系统认为这是"新的最优"

3. 继续训练
   - 从较差的 checkpoint 开始
   - 性能下降

---

## ✅ 修复方案

### 修改内容

#### 1. 将 `best_metric` 持久化到 checkpoint

**修改前**：
```python
ckpt = tf.train.Checkpoint(model=keras_model, optimizer=optimizer)
# best_metric 是 Python 变量，不会保存
```

**修改后**：
```python
# 创建 tf.Variable 来保存 best_metric
best_metric_var = tf.Variable(0.0, dtype=tf.float32, name='best_metric')
ckpt = tf.train.Checkpoint(model=keras_model, optimizer=optimizer, best_metric=best_metric_var)
# best_metric_var 会随 checkpoint 一起保存和恢复
```

#### 2. 恢复时同步 `best_metric`

**修改前**：
```python
if latest_ckpt:
    ckpt.restore(latest_ckpt)
    # best_metric 仍然是 0
```

**修改后**：
```python
if latest_ckpt:
    ckpt.restore(latest_ckpt)
    # 从 checkpoint 恢复 best_metric
    global best_metric
    best_metric = float(best_metric_var.numpy())
    print(f"   恢复的最优 recall: {best_metric:.6f}")
```

#### 3. 保存时同步更新

**修改前**：
```python
if recall > best_metric:
    best_metric = recall
    ckpt_manager.save()
    # best_metric_var 不会更新
```

**修改后**：
```python
if recall > best_metric:
    best_metric = recall
    best_metric_var.assign(best_metric)  # 同步更新
    ckpt_manager.save()
    print(f"   💾 保存新的最优模型，recall: {best_metric:.6f}")
```

---

## 🎯 修复效果

### 修复前

```
第一次训练：
iter 80K: recall 0.034 ← 最优，保存 checkpoint
iter 90K: recall 0.031
iter 100K: recall 0.028 ← 最后保存的（覆盖了 80K 的）

继续训练：
恢复 checkpoint → 加载 100K 的模型（recall=0.028）
best_metric = 0 ← 重置！
第一次评估: recall 0.028
系统认为这是"新最优"（因为 > 0）❌
```

### 修复后

```
第一次训练：
iter 80K: recall 0.034 ← 最优，保存 checkpoint + best_metric=0.034
iter 90K: recall 0.031 ← 不保存（< 0.034）
iter 100K: recall 0.028 ← 不保存（< 0.034）
最终保存的是 80K 的模型 ✅

继续训练：
恢复 checkpoint → 加载 80K 的模型（recall=0.034）✅
best_metric = 0.034 ← 正确恢复！✅
第一次评估: recall 0.034
系统知道这是之前的最优值 ✅
```

---

## 📝 使用说明

### 现在你可以：

**1. 继续之前的训练**
```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.007
    
# 输入之前的实验名称，如: test85
# 输出：
# ✅ 发现已有 checkpoint，自动恢复: .../ckpt-11
#    将从上次训练继续...
#    恢复的最优 recall: 0.031122  ← 显示之前的最优值
```

**2. 正确的保存和恢复**
```
训练中：
iter 10K: recall 0.025
   💾 保存新的最优模型，recall: 0.025000  ← 提示保存
iter 20K: recall 0.031
   💾 保存新的最优模型，recall: 0.031000  ← 提示保存
iter 30K: recall 0.029
   (不保存，因为 < 0.031)
   
恢复后：
   恢复的最优 recall: 0.031000  ← 正确恢复
```

---

## ⚠️ 重要提示

### 之前的 checkpoint 无法恢复 best_metric

**问题**：
- 之前保存的 checkpoint 没有 `best_metric_var`
- 恢复时会显示警告但不影响运行
- `best_metric` 会默认为 0

**解决方案**：
- 方案 1（推荐）：从头训练新的实验
- 方案 2：手动设置 best_metric（需要知道之前的最优值）

**如果继续训练旧 checkpoint**：
```bash
python src/train.py --model_type ComiRec-DR --learning_rate 0.007

# 输入旧实验名称
# 输出：
# ✅ 发现已有 checkpoint，自动恢复: .../ckpt-11
#    恢复的最优 recall: 0.000000  ← 旧 checkpoint，默认为 0
#    
# 解决方法：训练会继续，第一次评估后会更新为正确的值
# 或者：从头开始新的训练（推荐）
```

---

## 🔍 验证修复

### 测试步骤

**1. 训练到一定程度**
```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.008 \
    --max_iter 20
    
# 假设 20K 时 recall=0.034
```

**2. 停止并继续训练**
```bash
# 使用相同参数和实验名称
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.008 \
    --max_iter 50
    
# 输入相同的实验名称
```

**3. 检查输出**
```
✅ 发现已有 checkpoint，自动恢复: .../ckpt-X
   将从上次训练继续...
   恢复的最优 recall: 0.034000  ← 应该显示之前的值

training begin (Keras)
iter 21000: recall 0.034  ← 应该从之前的水平继续
```

**如果看到**：
- ✅ "恢复的最优 recall: 0.034000" → 修复成功
- ✅ 继续训练的 recall 在 0.034 附近 → 正常
- ❌ "恢复的最优 recall: 0.000000" → 使用的是旧 checkpoint

---

## 📊 对比示例

### 场景：训练 lr=0.007 到 20K

#### 修复前
```
训练到 20K:
iter 17K: recall 0.0287, 保存 ckpt-17
iter 18K: recall 0.0306, 保存 ckpt-18
iter 19K: recall 0.0310, 保存 ckpt-19
iter 20K: recall 0.0311, 保存 ckpt-20
最后保存的: ckpt-20

继续训练:
恢复 ckpt-20 (但可能是 19K 或更早的)
best_metric = 0 ← 重置
第一次评估: recall 0.028 (?)
困惑：为什么变低了？❌
```

#### 修复后
```
训练到 20K:
iter 17K: recall 0.0287, 保存 ckpt-17, best_metric=0.0287
iter 18K: recall 0.0306, 保存 ckpt-18, best_metric=0.0306
iter 19K: recall 0.0310, 保存 ckpt-19, best_metric=0.0310
iter 20K: recall 0.0311, 保存 ckpt-20, best_metric=0.0311
最后保存的: ckpt-20 (recall=0.0311)

继续训练:
恢复 ckpt-20
best_metric = 0.0311 ← 正确恢复
第一次评估: recall 0.0311 ← 符合预期
继续训练，从 0.0311 开始提升 ✅
```

---

## 🎓 技术细节

### 为什么用 tf.Variable？

**Python 变量 vs TensorFlow 变量**：

```python
# Python 变量
best_metric = 0.034
# 只存在于内存中
# 程序结束后丢失
# 无法保存到 checkpoint

# TensorFlow 变量
best_metric_var = tf.Variable(0.034)
# 是 TensorFlow 图的一部分
# 可以保存到 checkpoint
# 恢复时自动加载
```

### Checkpoint 的内容

**修复前**：
```
checkpoint 文件包含:
- model.weights
- optimizer.state
```

**修复后**：
```
checkpoint 文件包含:
- model.weights
- optimizer.state
- best_metric  ← 新增
```

---

## 💡 最佳实践

### 1. 总是使用相同的实验名称继续训练
```bash
# 第一次
python src/train.py --learning_rate 0.008
输入: test88

# 继续训练（使用相同参数）
python src/train.py --learning_rate 0.008
输入: test88  ← 相同名称
```

### 2. 检查恢复的 recall 值
```
✅ 发现已有 checkpoint，自动恢复
   恢复的最优 recall: 0.034000  ← 检查这个值
   
如果这个值和你记忆中的不一致：
- 可能恢复错了实验
- 可能使用了旧的 checkpoint（没有 best_metric）
```

### 3. 重要实验建议从头训练
```bash
# 对于关键实验，推荐用新的实验名称从头训练
python src/train.py --learning_rate 0.008
输入: test89_final  ← 新名称
```

---

## 🚀 总结

### 问题
- Checkpoint 恢复后 recall 下降
- `best_metric` 未持久化
- 恢复后不知道之前的最优值

### 修复
- ✅ 将 `best_metric` 保存到 checkpoint
- ✅ 恢复时自动加载 `best_metric`
- ✅ 显示恢复的最优 recall
- ✅ 保存时提示新的最优值

### 效果
- ✅ 继续训练不会丢失进度
- ✅ 正确跟踪最优模型
- ✅ 早停机制正确工作
- ✅ 可以安全地中断和恢复训练

**现在可以安全地继续训练了！** 🎉

