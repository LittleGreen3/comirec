# ComiRec 使用指南

## 训练模式选择

ComiRec-DR 和 ComiRec-SA 支持两种训练模式：

### 1. Keras/TF2 模式（推荐）✅

**使用方法**：
```bash
python src/train.py --model_type ComiRec-DR --use_keras
```

**特点**：
- ✅ 使用现代 TensorFlow 2.x Keras API
- ✅ 训练速度更快（使用 @tf.function 编译）
- ✅ 内存效率更高（使用 tf.train.Checkpoint）
- ✅ 代码更简洁易维护
- ✅ 已修复多兴趣向量评估逻辑，确保 recall 高于 DNN

**运行时提示**：
```
================================================================================
🚀 使用 Keras/TF2 训练模式
📊 模型类型: ComiRec-DR
💡 多兴趣模型: 评估时将使用所有 4 个兴趣向量
   - 训练: 使用单个readout向量计算loss
   - 评估: 使用所有兴趣向量搜索并合并结果，提高recall
================================================================================
```

### 2. TensorFlow 1.x 兼容模式

**使用方法**：
```bash
python src/train.py --model_type ComiRec-DR
# 不加 --use_keras 参数
```

**特点**：
- 使用 tf.compat.v1 API（兼容旧代码）
- 同样支持多兴趣向量评估
- 适合对比验证结果

**运行时提示**：
```
================================================================================
🔧 使用 TensorFlow 1.x 兼容模式
📊 模型类型: ComiRec-DR
💡 多兴趣模型: 评估时将使用所有 4 个兴趣向量
   - 训练: 使用单个readout向量计算loss
   - 评估: 使用所有兴趣向量搜索并合并结果，提高recall
================================================================================
```

## 完整训练示例

### 训练 ComiRec-DR（推荐 Keras 模式）
```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-DR \
    --use_keras \
    --learning_rate 0.001 \
    --embedding_dim 64 \
    --hidden_size 64 \
    --num_interest 4 \
    --max_iter 1000 \
    --patience 50
```

### 训练 ComiRec-SA（推荐 Keras 模式）
```bash
python src/train.py \
    --dataset book \
    --model_type ComiRec-SA \
    --use_keras \
    --learning_rate 0.001 \
    --embedding_dim 64 \
    --hidden_size 64 \
    --num_interest 4 \
    --max_iter 1000 \
    --patience 50
```

### 训练 DNN 作为基线对比
```bash
python src/train.py \
    --dataset book \
    --model_type DNN \
    --use_keras \
    --learning_rate 0.001 \
    --embedding_dim 64 \
    --hidden_size 64 \
    --max_iter 1000 \
    --patience 50
```

## 多兴趣向量评估策略

### 为什么 ComiRec recall 会高于 DNN？

**DNN 模型**：
- 1 个用户向量（历史 item 的平均）
- 搜索 1 次，获得 50 个候选 item
- 覆盖面有限

**ComiRec-DR 模型（num_interest=4）**：
- 4 个兴趣向量（通过胶囊网络学习）
- 搜索 4 次，每次获得 50 个候选 item
- 总计 200 个候选 item → 按相似度排序 → 去重 → 取前 50
- **覆盖面更广，更容易命中真实的 target item**

### 评估流程

1. **训练阶段**：
   - 输入：历史序列 + 目标 item
   - 输出：readout 向量（单个）
   - 计算 sampled softmax loss

2. **评估阶段**：
   - 输入：历史序列
   - 输出：所有兴趣向量（4 个）
   - 对每个兴趣向量搜索 top-50 item
   - 合并结果，去重，取 top-50
   - 计算 Recall@50, NDCG@50 等指标

## 预期结果

根据原始论文（TensorFlow 1.14 基准），在 Amazon Books 数据集上：

| 模型 | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 |
|------|-----------|---------|-----------|---------|
| DNN | 4.567 | 7.670 | 7.312 | 12.075 |
| ComiRec-DR | **5.311** | **9.185** | **8.106** | **13.520** |
| ComiRec-SA | **5.489** | **8.991** | **8.467** | **13.563** |

**注**：所有数字为百分比（%）。

## 故障排查

### 如果 ComiRec recall 仍然低于 DNN：

1. **确认使用了多兴趣向量评估**：
   - 检查日志中是否有 "💡 多兴趣模型" 提示
   - 确认评估时输出 shape 为 3 维：`[batch_size, num_interest, dim]`

2. **检查训练是否充分**：
   - loss 应该持续下降
   - 可能需要更多训练迭代

3. **尝试调整超参数**：
   - `num_interest`：尝试 4, 8
   - `learning_rate`：尝试 0.001, 0.0001
   - `embedding_dim`：尝试 64, 128

4. **对比两种模式的结果**：
   - 分别用 Keras 模式和 TF1.x 模式训练
   - 验证结果是否一致

## 参考

- 原始论文：[Controllable Multi-Interest Framework for Recommendation](https://arxiv.org/abs/2005.09347)
- 修复文档：`CRITICAL_FIX_SUMMARY.md`
- 问题分析：`ANALYSIS_COMIREC_ISSUES.md`

