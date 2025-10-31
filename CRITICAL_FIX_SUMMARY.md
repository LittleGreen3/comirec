# ComiRec-DR Recall 低于 DNN 的根本原因及修复

## 核心问题

**Keras版本和TF1.x兼容模式的评估策略不一致！**

### TF1.x兼容模式（正确）
- ComiRec-DR 的 `output_user` 返回 `user_eb`（所有兴趣向量）
- 形状：`[batch_size, num_interest, dim]`
- 评估时：对每个兴趣向量分别搜索，然后合并结果
- **效果**：能够覆盖用户的多个兴趣，recall 高

### Keras版本（有问题）
- ComiRec-DR 的 `call` 返回 `readout`（单个向量）
- 形状：`[batch_size, dim]`
- 评估时使用 `keras_model([dummy_mid, hist_item, hist_mask])`，调用 `call` 方法
- 只返回一个向量，只能搜索一次
- **效果**：只能覆盖一个兴趣，recall 低

## 修复方案

### 1. 修改 Keras 版本的 ComiRec-DR 和 ComiRec-SA

添加 `output_user_interests` 方法，返回所有兴趣向量：

```python
class KerasModelComiRecDR(KerasModelBase):
    def call(self, inputs, training=False):
        # ... 原有逻辑 ...
        # 训练时返回readout用于计算loss
        return readout, item_emb
    
    def output_user_interests(self, mid_hist, mask):
        """返回所有兴趣向量用于评估（与TF1.x版本一致）"""
        # 使用dummy item来计算interest_capsule
        batch_size = tf.shape(mid_hist)[0]
        dummy_mid = tf.zeros((batch_size,), dtype=tf.int32)
        item_emb = self.embed_items(dummy_mid)
        hist_emb = self.embed_items(mid_hist)
        interest_capsule, _ = self.capsule(hist_emb, item_emb, mask)
        return interest_capsule  # [batch_size, num_interest, dim]

    def output_user(self, mid_hist, mask):
        # 评估时返回所有兴趣向量（与TF1.x版本一致）
        return self.output_user_interests(mid_hist, mask)
```

### 2. 修改 Keras 版本的评估函数

使用 `output_user` 方法而不是 `call` 方法：

```python
# 旧代码（错误）
user_vec, _ = keras_model([
    tf.convert_to_tensor(dummy_mid, dtype=tf.int32),
    tf.convert_to_tensor(hist_item, dtype=tf.int32),
    tf.convert_to_tensor(hist_mask, dtype=tf.float32)
], training=False)  # 只返回单个向量

# 新代码（正确）
user_embs = keras_model.output_user(
    tf.convert_to_tensor(hist_item, dtype=tf.int32),
    tf.convert_to_tensor(hist_mask, dtype=tf.float32)
).numpy()  # 返回所有兴趣向量

# 根据维度判断是否为多兴趣模型
if len(user_embs.shape) == 3:
    # 多兴趣模型：对每个兴趣向量分别搜索，然后合并结果
    # ... 多兴趣向量处理逻辑 ...
else:
    # 单向量模型：直接搜索
    # ... 单向量处理逻辑 ...
```

## 关键改进点

1. **训练和评估分离**：
   - 训练时：`call` 方法返回 `readout`（单个向量），用于计算 loss
   - 评估时：`output_user` 方法返回所有兴趣向量（多个向量），用于搜索

2. **保持一致性**：
   - Keras 版本现在与 TF1.x 兼容模式使用相同的评估策略
   - 都使用多兴趣向量进行评估

3. **多兴趣向量合并策略**：
   - 对每个兴趣向量分别搜索 topN 个item
   - 将所有结果按相似度排序
   - 去重后取前 topN 个

## 为什么这样能确保 ComiRec recall 高于 DNN？

### DNN 模型
- 只有一个用户向量（历史 item 的平均）
- 只能搜索一次
- 只能捕获用户的一个主要兴趣

### ComiRec-DR 模型
- 有 4 个兴趣向量（默认 `num_interest=4`）
- 每个兴趣向量搜索一次，共搜索 4 次
- 能够捕获用户的多个不同兴趣
- 合并结果后，覆盖面更广，recall 更高

## 预期效果

修复后：
- **ComiRec-DR 的 recall 应该显著高于 DNN**（根据 TF1.14 基准，应该提升 0.7-0.8 个百分点）
- **NDCG 也应该相应提升**（因为能够推荐更多相关的 item）
- **与原始 TensorFlow 1.14 实现的结果对齐**

## 代码位置

- `src/model.py`:
  - `KerasModelComiRecDR.output_user_interests` (第489-497行)
  - `KerasModelComiRecDR.output_user` (第499-501行)
  - `KerasModelComiRecSA.output_user_interests` (第533-544行)
  - `KerasModelComiRecSA.output_user` (第546-548行)

- `src/train.py`:
  - `evaluate_full_keras` 修改 (第242-317行)

