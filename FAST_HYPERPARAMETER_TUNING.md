# ComiRec-DR 快速超参数调优指南

## 🎯 核心问题

**问题**：ComiRec-DR 训练太慢（3-4小时），如何快速验证参数有效性？

**解决方案**：通过早期指标（前 30-50K 迭代）预测最终效果

---

## ⚡ 快速验证策略

### 策略 1: 观察前 50K 迭代的表现 ⭐⭐⭐

**原理**：
- 好的超参数在**早期**就会显示出优势
- 学习率合适 → loss 下降快 → 早期 recall 高
- 学习率不合适 → loss 下降慢 → 早期 recall 低

**判断标准**：

| 迭代数 | 观察指标 | 好的配置 | 差的配置 |
|--------|---------|---------|---------|
| **10K** | train loss | < 7.5 | > 8.0 |
| **20K** | train loss | < 6.5 | > 7.5 |
| **30K** | train loss | < 6.0 | > 7.0 |
| **50K** | valid recall | > 0.04 | < 0.03 |
| **50K** | valid recall | > 0.05 | < 0.04 |

**实践**：
```bash
# 只跑 50K 迭代（约 30-40 分钟）
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --max_iter 50 \
    --test_iter 10
```

**判断**：
- 如果 50K 时 recall > 0.05，继续训练
- 如果 50K 时 recall < 0.03，参数有问题，停止

---

### 策略 2: 并行运行多个实验 ⭐⭐

**原理**：同时测试多个参数，快速比较

**方法 A：使用多个终端**
```bash
# 终端 1: lr=0.003
python src/train.py --model_type ComiRec-DR --learning_rate 0.003 --max_iter 50

# 终端 2: lr=0.005
python src/train.py --model_type ComiRec-DR --learning_rate 0.005 --max_iter 50

# 终端 3: lr=0.007
python src/train.py --model_type ComiRec-DR --learning_rate 0.007 --max_iter 50
```

**方法 B：使用脚本自动化**
```bash
# 创建一个测试脚本
for lr in 0.003 0.005 0.007; do
    echo "Testing lr=$lr"
    python src/train.py \
        --model_type ComiRec-DR \
        --learning_rate $lr \
        --max_iter 50 \
        --test_iter 10 &
done
wait
```

**比较**：看哪个在 50K 时 recall 最高

---

### 策略 3: 使用更小的数据集 ⭐⭐

**原理**：在小数据上快速验证，然后在全量数据上精调

**实现**：
```python
# 修改 data_iterator.py，只使用 10% 的用户
class DataIterator:
    def read(self, source):
        # ... 原有代码 ...
        # 添加：只使用部分数据
        if hasattr(self, 'sample_ratio'):
            num_users = int(len(self.users) * self.sample_ratio)
            self.users = self.users[:num_users]
```

**使用**：
```bash
# 在 10% 数据上快速测试（速度提升 10 倍）
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --max_iter 100 \
    --sample_ratio 0.1
```

**注意**：
- 小数据上的绝对值可能不同
- 但**相对趋势**是一致的
- 哪个参数在小数据上好，在大数据上也好

---

### 策略 4: 学习曲线分析 ⭐⭐⭐

**原理**：通过前期曲线斜率预测收敛点

**关键指标**：

#### 指标 1: Loss 下降速度
```
好的配置：
iter 10K: loss 7.5 → iter 20K: loss 6.5  (下降 1.0)
iter 20K: loss 6.5 → iter 30K: loss 6.0  (下降 0.5)
→ 下降快，说明学习率合适

差的配置：
iter 10K: loss 8.0 → iter 20K: loss 7.8  (下降 0.2)
iter 20K: loss 7.8 → iter 30K: loss 7.6  (下降 0.2)
→ 下降慢，说明学习率太小
```

#### 指标 2: Recall 增长速度
```
好的配置：
iter 10K: recall 0.020 → iter 30K: recall 0.040 → iter 50K: recall 0.055
→ 持续增长，预计最终 > 0.080

差的配置：
iter 10K: recall 0.015 → iter 30K: recall 0.025 → iter 50K: recall 0.030
→ 增长缓慢，预计最终 < 0.065
```

#### 指标 3: Loss vs Recall 的关系
```
好的配置：
loss 下降 → recall 同步上升
→ 模型在正确学习

差的配置：
loss 下降 → recall 几乎不变
→ 可能过拟合或参数不对
```

---

## 📊 快速判断标准

### 30 分钟判断法（50K 迭代）

**步骤 1**: 运行到 50K 迭代（~30-40 分钟）

**步骤 2**: 检查以下指标

| 指标 | 优秀 | 良好 | 一般 | 差 |
|------|------|------|------|-----|
| **Loss (50K)** | < 5.5 | 5.5-6.0 | 6.0-6.5 | > 6.5 |
| **Recall (50K)** | > 0.06 | 0.05-0.06 | 0.04-0.05 | < 0.04 |
| **Loss 下降速度** | 快 | 中 | 慢 | 极慢 |
| **Recall 增长趋势** | 持续 | 稳定 | 波动 | 停滞 |

**步骤 3**: 做决策

- ✅ **优秀/良好**：继续训练到收敛（预计 150-200K）
- ⚠️ **一般**：可以继续，但可能不是最优
- ❌ **差**：停止，调整参数重新开始

---

## 🧪 实验设计建议

### 阶段 1: 快速筛选学习率（1-2 小时）

**目标**：找到最优学习率

```bash
# 并行测试 3 个学习率，每个跑 50K
for lr in 0.003 0.005 0.007; do
    python src/train.py \
        --model_type ComiRec-DR \
        --learning_rate $lr \
        --max_iter 50 \
        --test_iter 10 \
        2>&1 | tee log_lr_${lr}.txt &
done
```

**预期时间**：30-40 分钟（并行）

**选择标准**：50K 时 recall 最高的学习率

---

### 阶段 2: 测试负样本数（1 小时）

**目标**：在最优学习率下测试 neg_num

```bash
# 使用阶段 1 找到的最优 lr（假设是 0.005）
for neg in 10 20 30; do
    python src/train.py \
        --model_type ComiRec-DR \
        --learning_rate 0.005 \
        --neg_num $neg \
        --max_iter 50 \
        --test_iter 10 \
        2>&1 | tee log_neg_${neg}.txt &
done
```

**预期时间**：30-40 分钟（并行）

---

### 阶段 3: 测试兴趣数量（1 小时）

**目标**：测试不同的 num_interest

```bash
for num in 4 8; do
    python src/train.py \
        --model_type ComiRec-DR \
        --learning_rate 0.005 \
        --neg_num 20 \
        --num_interest $num \
        --max_iter 50 \
        --test_iter 10 \
        2>&1 | tee log_interest_${num}.txt &
done
```

**预期时间**：30-40 分钟

---

### 阶段 4: 完整训练（3-4 小时）

**目标**：用最优参数训练到收敛

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --neg_num 20 \
    --num_interest 4 \
    --patience 100 \
    --max_iter 1000
```

**总耗时**：1-2h（筛选）+ 3-4h（完整训练）= **4-6 小时**

对比：盲目尝试需要 10-20 小时

---

## 🔍 关键观察点

### 前 10K 迭代（~6 分钟）

**观察**：
- Train loss 是否在快速下降？
- 第一次评估（10K）的 recall 是多少？

**判断**：
```
优秀: loss < 7.5, recall > 0.025
良好: loss 7.5-8.0, recall 0.020-0.025
差: loss > 8.0, recall < 0.020
```

**行动**：
- 如果差：立即停止，调整学习率

---

### 前 30K 迭代（~20 分钟）

**观察**：
- Loss 下降曲线是否平滑？
- Recall 是否在持续增长？

**判断**：
```
优秀: loss < 6.0, recall > 0.045
良好: loss 6.0-6.5, recall 0.035-0.045
差: loss > 6.5, recall < 0.035
```

**行动**：
- 如果良好以上：继续
- 如果差：停止

---

### 前 50K 迭代（~30-40 分钟）

**观察**：
- Recall 的增长速度
- 与 DNN 的对比

**判断**：
```
优秀: recall > 0.060 (接近 DNN)
良好: recall 0.050-0.060
一般: recall 0.040-0.050
差: recall < 0.040
```

**行动**：
- 优秀：肯定会超过 DNN，继续训练
- 良好：可能持平 DNN，继续训练
- 一般：可能不如 DNN，考虑调参
- 差：肯定不如 DNN，停止并调参

---

## 📈 学习率对比示例

### 示例数据（50K 迭代）

| Learning Rate | Loss (50K) | Recall (50K) | 预测最终 Recall | 决策 |
|--------------|-----------|-------------|---------------|------|
| **0.003** | 6.5 | 0.045 | ~0.070 | ⚠️ 偏低 |
| **0.005** | 5.8 | 0.058 | ~0.085 | ✅ 最优 |
| **0.007** | 6.0 | 0.052 | ~0.078 | ⚠️ 可能震荡 |
| **0.010** | 7.2 | 0.035 | ~0.060 | ❌ 太大，不稳定 |

**结论**：选择 lr=0.005

---

## 🎓 经验规则

### 规则 1: 10K 迭代能看出学习率问题

```
lr 太小: loss 下降极慢，可能还在 8.0+
lr 合适: loss 快速下降到 7.5 左右
lr 太大: loss 震荡，或上升
```

**判断时间**：6-8 分钟

---

### 规则 2: 30K 迭代能看出收敛趋势

```
好的配置: recall 持续增长 (0.02 → 0.03 → 0.04)
差的配置: recall 停滞 (0.02 → 0.022 → 0.025)
```

**判断时间**：20 分钟

---

### 规则 3: 50K 迭代能预测最终效果

```
经验公式：
最终 Recall ≈ Recall(50K) × 1.4 ~ 1.5

例如：
Recall(50K) = 0.058 → 最终 ≈ 0.081 ~ 0.087
Recall(50K) = 0.035 → 最终 ≈ 0.049 ~ 0.053
```

**判断时间**：30-40 分钟

---

## 💡 实用技巧

### 技巧 1: 使用 TensorBoard 实时监控

```bash
# 启动 TensorBoard
tensorboard --logdir runs/

# 在浏览器打开
http://localhost:6006
```

**优势**：
- 实时查看多个实验的曲线
- 对比不同参数的效果
- 无需等待训练结束

---

### 技巧 2: 设置更频繁的评估（早期）

修改 `train.py`：
```python
# 前 50K 迭代，每 5K 评估一次
# 之后每 10K 评估一次
if iter < 50000:
    test_iter = 5000
else:
    test_iter = 10000
```

**优势**：更早发现问题

---

### 技巧 3: 记录关键指标到文件

```python
# 在评估时记录
with open(f'quick_eval_{exp_name}.txt', 'a') as f:
    f.write(f"{iter}\t{loss:.4f}\t{recall:.6f}\t{ndcg:.6f}\n")
```

**优势**：快速比较多个实验

---

### 技巧 4: 使用 Loss 作为早期指标

```python
# 如果 30K 迭代时 loss 还 > 6.5，自动停止
if iter == 30000 and current_loss > 6.5:
    print("Loss too high at 30K, stopping...")
    break
```

**优势**：节省无效实验时间

---

## 🚀 快速验证脚本

### 创建快速测试脚本

```python
# quick_test.py
import subprocess
import time

# 要测试的学习率
learning_rates = [0.003, 0.005, 0.007]

# 存储结果
results = {}

for lr in learning_rates:
    print(f"\n{'='*60}")
    print(f"Testing lr={lr}")
    print('='*60)
    
    start = time.time()
    
    # 运行 50K 迭代
    cmd = f"""python src/train.py \
        --model_type ComiRec-DR \
        --learning_rate {lr} \
        --max_iter 50 \
        --test_iter 10"""
    
    subprocess.run(cmd, shell=True)
    
    elapsed = time.time() - start
    print(f"Time: {elapsed/60:.1f} minutes")
    
    # TODO: 解析日志文件，提取最终 recall
    # results[lr] = parse_log(...)

# 打印对比结果
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for lr, recall in results.items():
    print(f"lr={lr}: recall={recall:.6f}")
```

---

## 📊 决策树

```
开始训练
    │
    ├─ 10K 迭代（6分钟）
    │   ├─ loss > 8.0？ → 停止，增大学习率
    │   └─ loss < 7.5？ → 继续
    │
    ├─ 30K 迭代（20分钟）
    │   ├─ recall < 0.035？ → 停止，调整参数
    │   ├─ recall 0.035-0.045？ → 继续观察
    │   └─ recall > 0.045？ → 很好，继续
    │
    ├─ 50K 迭代（30-40分钟）
    │   ├─ recall < 0.045？ → 停止，这个配置不行
    │   ├─ recall 0.045-0.055？ → 可以继续但不是最优
    │   └─ recall > 0.055？ → 优秀，训练到收敛
    │
    └─ 继续训练到收敛（150-200K）
```

---

## ✅ 最佳实践总结

### 1. 分阶段验证
- ✅ 第 1 阶段（10K）：检查学习率是否合理
- ✅ 第 2 阶段（30K）：检查收敛趋势
- ✅ 第 3 阶段（50K）：预测最终效果
- ✅ 第 4 阶段（150-200K）：完整训练

### 2. 并行实验
- ✅ 同时测试多个参数
- ✅ 使用多个终端或 tmux
- ✅ 总时间 = 单个实验时间（而非 N 倍）

### 3. 关键指标
- ✅ Loss 下降速度（前 10K）
- ✅ Recall 增长速度（前 30K）
- ✅ 50K 时的绝对值

### 4. 经验规则
- ✅ 50K recall > 0.055 → 继续训练
- ✅ 50K recall < 0.045 → 停止并调参
- ✅ Loss 下降慢 → 增大学习率
- ✅ Loss 震荡 → 减小学习率

---

## 🎯 立即行动方案

### 步骤 1: 快速测试（40 分钟）

```bash
# 同时测试 3 个学习率
python src/train.py --model_type ComiRec-DR --learning_rate 0.003 --max_iter 50 &
python src/train.py --model_type ComiRec-DR --learning_rate 0.005 --max_iter 50 &
python src/train.py --model_type ComiRec-DR --learning_rate 0.007 --max_iter 50 &
```

### 步骤 2: 观察结果

在 `runs/` 目录查看日志，比较 50K 时的 recall

### 步骤 3: 选择最优参数

选择 50K recall 最高的配置

### 步骤 4: 完整训练

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate <最优lr> \
    --patience 100 \
    --max_iter 1000
```

**总时间**：40 分钟（快速测试）+ 3 小时（完整训练）= **~4 小时**

比盲目尝试节省 **50% 时间**！🚀

