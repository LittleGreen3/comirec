# Metrics@N 配置说明

## 📌 核心概念

**Metrics@N** 中的 **N** 代表推荐列表的长度（top-N），即模型给用户推荐多少个物品。

- **Metrics@20**：评估推荐列表前 20 个物品的表现
- **Metrics@50**：评估推荐列表前 50 个物品的表现

## 🔧 配置位置

### 1. 命令行参数定义

**位置**：`src/train.py` 第 70 行

```python
parser.add_argument('--topN', type=int, default=50)
```

**说明**：
- 参数名：`--topN`
- 默认值：`50`（即默认评估 Metrics@50）
- 类型：整数

### 2. 如何修改为 Metrics@20

**方法一：通过命令行参数修改**

```bash
python src/train.py -p train --topN 20 --model_type ComiRec-DR --dataset book
```

**方法二：修改代码默认值**

编辑 `src/train.py` 第 70 行：
```python
parser.add_argument('--topN', type=int, default=20)  # 改为 20
```

### 3. topN 在代码中的使用位置

#### 位置 1：评估函数参数定义
```70:70:src/train.py
parser.add_argument('--topN', type=int, default=50)
```

#### 位置 2：传递给评估函数
```python
# 第 536 行（训练时评估）
metrics = evaluate_full_keras(valid_data, keras_model, item_cate_map, 
                              args.topN, args.embedding_dim, model_type=model_type)

# 第 581, 585 行（最终评估）
metrics = evaluate_full_keras(valid_data, keras_model, item_cate_map, 
                              args.topN, args.embedding_dim, model_type=model_type, save=False)
```

#### 位置 3：在评估函数中使用
```331:331:src/train.py
D, I = gpu_index.search(user_vec, topN)
```

**关键逻辑**：
- `gpu_index.search(user_vec, topN)` 从 faiss 索引中搜索与 `user_vec` 最相似的 **topN 个物品**
- 返回的 `I` 数组包含 topN 个物品的索引
- 后续计算 recall、ndcg 等指标时，都基于这 topN 个推荐物品

### 4. 评估指标计算流程

```
1. 用户向量 (user_vec) 
   ↓
2. Faiss 搜索 topN 个最相似物品
   ↓
3. 计算指标：
   - Recall@N: 在 topN 个推荐中，有多少个是真实标签
   - NDCG@N: 归一化折损累积增益（考虑位置权重）
   - HitRate@N: 是否至少命中一个真实标签
```

## 📊 示例

### 示例 1：评估 Metrics@20

```bash
python src/train.py -p train \
    --model_type ComiRec-DR \
    --dataset book \
    --topN 20 \
    --learning_rate 0.001 \
    --max_iter 1000
```

输出示例：
```
iter: 1000, train loss: 6.5234, valid recall: 0.023456, valid ndcg: 0.017890, valid hitrate: 0.052341
```

### 示例 2：评估 Metrics@50（默认）

```bash
python src/train.py -p train \
    --model_type ComiRec-DR \
    --dataset book \
    --topN 50 \
    --learning_rate 0.001 \
    --max_iter 1000
```

输出示例：
```
iter: 1000, train loss: 6.5234, valid recall: 0.028901, valid ndcg: 0.019456, valid hitrate: 0.062341
```

**注意**：通常 Metrics@50 的 recall 会比 Metrics@20 更高（因为推荐列表更长），但计算成本也更高。

## 🔍 代码流程图

```
命令行参数 (--topN)
    ↓
args.topN (默认 50)
    ↓
evaluate_full_keras(..., topN=args.topN, ...)
    ↓
gpu_index.search(user_vec, topN)  # 搜索 topN 个物品
    ↓
计算 Recall@N, NDCG@N, HitRate@N
```

## 💡 建议

1. **实验阶段**：使用较小的 topN（如 10 或 20）可以更快地迭代实验
2. **最终评估**：使用标准的 topN（如 20、50）与其他论文对比
3. **生产环境**：根据实际业务需求选择 topN（通常 10-50 之间）

## 📝 总结

- **配置位置**：`src/train.py` 第 70 行 `--topN` 参数
- **默认值**：50（Metrics@50）
- **修改方法**：命令行参数 `--topN 20` 或修改代码默认值
- **影响范围**：所有评估指标（Recall、NDCG、HitRate）

