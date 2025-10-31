# 实用的快速参数测试方案（避免资源冲突）

## ⚠️ 并行运行的资源占用问题

### 问题分析

**同时运行 3 个训练进程会：**
- ✅ GPU 内存：每个进程可能占用 2-4GB → 总共 6-12GB
- ❌ 如果 GPU 只有 8GB：会 OOM（内存溢出）
- ❌ CPU 和 I/O 竞争：数据加载变慢
- ❌ 速度反而变慢：不是 3 倍快，可能是 4-5 倍慢

**结论**：不建议并行运行，除非你有：
- 32GB+ GPU 显存，或
- 多块 GPU

---

## ✅ 推荐方案：串行快速测试

### 方案 1: 超快速单次测试（最佳）⭐⭐⭐

**只跑 20K 迭代**（约 12-15 分钟），就能判断学习率是否合适！

```bash
# 测试 1: lr=0.003
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.003 \
    --max_iter 20 \
    --test_iter 10

# 看结果，如果不好就直接跳过下一个
# 测试 2: lr=0.005
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --max_iter 20 \
    --test_iter 10

# 测试 3: lr=0.007
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.007 \
    --max_iter 20 \
    --test_iter 10
```

**判断标准**（20K 迭代）：

| Loss (20K) | Recall (20K) | 判断 |
|-----------|-------------|------|
| **< 6.5** | **> 0.038** | ✅ 很好，继续训练 |
| **6.5-7.0** | **0.030-0.038** | ⚠️ 一般，可以考虑 |
| **> 7.0** | **< 0.030** | ❌ 太差，跳过 |

**总时间**：
- 如果第一个就很好：15 分钟
- 测试 2 个：30 分钟
- 测试全部 3 个：45 分钟

---

### 方案 2: 更短但更保守（10 分钟快速筛查）

**只跑 10K 迭代**（约 6-8 分钟），快速排除明显不好的参数：

```bash
# 快速筛查，如果 loss > 7.5 或 recall < 0.025，直接跳过
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.003 \
    --max_iter 10 \
    --test_iter 5
```

**判断标准**（10K 迭代）：

| Loss (10K) | Recall (10K) | 判断 |
|-----------|-------------|------|
| **< 7.5** | **> 0.025** | ✅ 继续测试到 20K |
| **7.5-8.0** | **0.020-0.025** | ⚠️ 勉强，可以继续 |
| **> 8.0** | **< 0.020** | ❌ 立即停止，尝试下一个 |

**流程**：
1. 10K 测试（6-8 分钟）→ 如果好，继续
2. 20K 测试（再加 6-8 分钟）→ 最终判断
3. 选择最优，完整训练

**总时间**：
- 最快：10 分钟（第一个就很好）
- 平均：20-30 分钟（测试 2-3 个参数）
- 最长：45 分钟（全部测试）

---

### 方案 3: 逐步缩小范围（最精确）

**第一阶段**：快速扫描（10 分钟 × 5 = 50 分钟）

```bash
# 测试 5 个学习率，每个只跑 10K
for lr in 0.002 0.003 0.005 0.007 0.010; do
    python src/train.py \
        --model_type ComiRec-DR \
        --learning_rate $lr \
        --max_iter 10 \
        --test_iter 5
done
```

**选择 2 个最好的**，进入第二阶段

**第二阶段**：精确测试（20 分钟 × 2 = 40 分钟）

```bash
# 只测试第一阶段最好的 2 个
python src/train.py --model_type ComiRec-DR --learning_rate <best1> --max_iter 50
python src/train.py --model_type ComiRec-DR --learning_rate <best2> --max_iter 50
```

**第三阶段**：完整训练（3 小时）

**总时间**：50min + 40min + 3h = **~5 小时**

---

## 🎯 最推荐：智能快速测试（20K 迭代）

### 为什么 20K 就够？

**根据经验**：
- **10K**: 能看出学习率是否"明显错误"
- **20K**: 能判断学习率是否"合适"
- **50K**: 能预测最终效果（但需要 3 倍时间）

**20K 迭代的优势**：
- ✅ 时间短：12-15 分钟
- ✅ 判断准：能识别 90% 的情况
- ✅ 资源省：串行运行，不冲突

---

### 实际操作步骤

#### 步骤 1: 快速测试（15 分钟 × 3 = 45 分钟）

```bash
# 测试 1: lr=0.005（最可能的值）
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --max_iter 20 \
    --test_iter 10

# 查看结果：loss 和 recall
# 如果 loss < 6.5 且 recall > 0.038 → 很好，直接完整训练
# 如果不够好，继续测试下一个
```

```bash
# 测试 2: lr=0.003（如果 0.005 太大）
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.003 \
    --max_iter 20 \
    --test_iter 10
```

```bash
# 测试 3: lr=0.007（如果 0.005 太小）
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.007 \
    --max_iter 20 \
    --test_iter 10
```

#### 步骤 2: 选择最优，完整训练（3 小时）

```bash
python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate <最优lr> \
    --patience 100 \
    --max_iter 1000
```

**总时间**：45 分钟（测试）+ 3 小时（训练）= **~4 小时**

---

## 📊 判断标准总结

### 10K 迭代（6-8 分钟）

| Loss (10K) | Recall (10K) | 决策 |
|-----------|-------------|------|
| < 7.5 | > 0.025 | ✅ 继续到 20K |
| 7.5-8.0 | 0.020-0.025 | ⚠️ 继续到 20K |
| > 8.0 | < 0.020 | ❌ 停止，下一个 |

### 20K 迭代（12-15 分钟）

| Loss (20K) | Recall (20K) | 预测最终 | 决策 |
|-----------|-------------|---------|------|
| < 6.5 | > 0.038 | > 0.08 | ✅ 完整训练 |
| 6.5-7.0 | 0.030-0.038 | 0.07-0.08 | ⚠️ 可以训练 |
| > 7.0 | < 0.030 | < 0.07 | ❌ 调整参数 |

### 50K 迭代（30-40 分钟）

| Recall (50K) | 预测最终 | 决策 |
|-------------|---------|------|
| > 0.058 | > 0.08 | ✅ 肯定超过 DNN |
| 0.050-0.058 | 0.07-0.08 | ⚠️ 可能超过 |
| < 0.050 | < 0.07 | ❌ 不如 DNN |

---

## 💡 实用技巧

### 技巧 1: 按可能性排序测试

**最可能好的参数先测试**：

```bash
# 1. 先测试 lr=0.005（最可能的值）
# 2. 如果不好，测试 lr=0.003（可能太小）
# 3. 如果还不好，测试 lr=0.007（可能太大）
```

**优势**：通常第一个就对了，节省时间

---

### 技巧 2: 创建快速测试脚本

```python
# quick_test_lr.py
import subprocess
import sys

learning_rates = [0.005, 0.003, 0.007]  # 按可能性排序

for lr in learning_rates:
    print(f"\n{'='*60}")
    print(f"测试 learning_rate = {lr}")
    print('='*60)
    
    cmd = f"""python src/train.py \
        --model_type ComiRec-DR \
        --learning_rate {lr} \
        --max_iter 20 \
        --test_iter 10"""
    
    result = subprocess.run(cmd, shell=True)
    
    # 询问是否继续
    if result.returncode == 0:
        user_input = input("\n是否继续测试下一个参数？(y/n): ")
        if user_input.lower() != 'y':
            break
    
print("\n测试完成！请查看结果并选择最优参数。")
```

**使用**：
```bash
python quick_test_lr.py
```

---

### 技巧 3: 只测试关键参数

**优先级排序**：

1. **learning_rate** ⭐⭐⭐（最关键，必须测试）
2. **neg_num** ⭐⭐（重要，但可以先固定）
3. **num_interest** ⭐（可选，影响不大）

**建议**：
- 第一步：只测试 learning_rate（20K 迭代 × 3 = 45 分钟）
- 第二步：用最优 lr，测试 neg_num（20K 迭代 × 2 = 30 分钟）
- 第三步：完整训练（3 小时）

**总时间**：45min + 30min + 3h = **~4.5 小时**

---

### 技巧 4: 记录结果便于对比

**在测试时记录**：

```bash
# 每次测试后，手动记录到文件
echo "lr=0.005, loss=6.2, recall=0.042" >> quick_test_results.txt
echo "lr=0.003, loss=6.8, recall=0.035" >> quick_test_results.txt
echo "lr=0.007, loss=6.0, recall=0.040" >> quick_test_results.txt
```

**查看对比**：
```bash
cat quick_test_results.txt
```

---

## 🔍 如果有多个 GPU

如果你有多个 GPU（2+），可以并行：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.003 \
    --max_iter 20 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.005 \
    --max_iter 20 &

# GPU 2
CUDA_VISIBLE_DEVICES=2 python src/train.py \
    --model_type ComiRec-DR \
    --learning_rate 0.007 \
    --max_iter 20 &

wait  # 等待所有完成
```

**前提**：确认你有多个 GPU

---

## 📋 推荐工作流

### 最省时方案（推荐）

```
1. 测试 lr=0.005（20K，15 分钟）
   ├─ 如果好 → 直接完整训练（3 小时）
   └─ 如果不好 → 继续
   
2. 测试 lr=0.003（20K，15 分钟）
   ├─ 如果好 → 完整训练（3 小时）
   └─ 如果不好 → 继续
   
3. 测试 lr=0.007（20K，15 分钟）
   └─ 选择最好的，完整训练（3 小时）
```

**预期时间**：
- 最快：15min（第一个就对了）+ 3h = **3.25 小时**
- 平均：45min（测试）+ 3h = **3.75 小时**
- 最长：45min（全部测试）+ 3h = **3.75 小时**

---

### 最保守方案（如果时间充裕）

```
1. 快速扫描 5 个 lr（10K × 5 = 50 分钟）
2. 精确测试前 2 个（50K × 2 = 80 分钟）
3. 完整训练最优（3 小时）
```

**总时间**：~5.5 小时

---

## ✅ 总结

### 推荐方案

**方案：串行快速测试（20K 迭代）**

```bash
# 按顺序测试 3 个学习率，每个 20K 迭代（12-15 分钟）
python src/train.py --model_type ComiRec-DR --learning_rate 0.005 --max_iter 20
python src/train.py --model_type ComiRec-DR --learning_rate 0.003 --max_iter 20
python src/train.py --model_type ComiRec-DR --learning_rate 0.007 --max_iter 20
```

**优势**：
- ✅ 不占用太多资源（串行）
- ✅ 速度快（每个 15 分钟）
- ✅ 判断准确（20K 足够判断）
- ✅ 避免内存冲突

**总时间**：45 分钟（测试）+ 3 小时（完整训练）= **~4 小时**

---

### 对比

| 方案 | 测试时间 | 资源占用 | 准确性 | 推荐度 |
|------|---------|---------|--------|--------|
| **并行 3 个（50K）** | 40 分钟 | 高（可能 OOM）| 高 | ❌ |
| **串行 3 个（20K）** | 45 分钟 | 低 | 高 | ✅✅✅ |
| **串行 1 个（50K）** | 40 分钟 | 低 | 高 | ✅✅ |
| **串行 5 个（10K）** | 50 分钟 | 低 | 中 | ⚠️ |

---

**立即开始**：按顺序测试 3 个学习率，每个只跑 20K 迭代！🚀

