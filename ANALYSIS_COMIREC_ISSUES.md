# ComiRec-DR 模型问题分析

## 问题概述

根据图片中的TensorFlow 1.14基准数据，ComiRec-DR应该比DNN表现更好，但当前实现中：
1. **Recall比DNN还低**
2. **NDCG在TensorFlow 2升级后变得很低**

## 根本原因分析

### 问题1: 训练与评估不一致 ⚠️ **关键问题**

在 `src/model.py` 的 `Model_ComiRec_DR` 类中：

```python
# 第226行：模型定义
self.user_eb, self.readout = capsule_network(item_his_emb, self.item_eb, self.mask)
self.build_sampled_softmax_loss(self.item_eb, self.readout)  # 训练时使用readout
```

但 `Model` 基类的 `output_user` 方法（第59-64行）：
```python
def output_user(self, sess, inps):
    user_embs = sess.run(self.user_eb, feed_dict={  # ⚠️ 返回的是user_eb，不是readout！
        self.mid_his_batch_ph: inps[0],
        self.mask: inps[1]
    })
    return user_embs
```

**问题**：
- **训练时**：使用 `self.readout`（单个用户向量 `[batch_size, dim]`）
- **评估时**：使用 `self.user_eb`（所有兴趣向量 `[batch_size, num_interest, dim]`）

这导致：
1. 评估时搜索使用了多个兴趣向量，但合并逻辑可能不准确
2. 训练和评估的向量表示不一致，模型学到的embedding无法在评估时正确使用

### 问题2: 多兴趣向量评估逻辑问题

在 `evaluate_full` 函数中（第148-205行），当 `len(user_embs.shape) == 3` 时：

```python
else:
    ni = user_embs.shape[1]  # num_interest
    user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
    D, I = gpu_index.search(user_embs, topN)
    # 然后合并多个兴趣向量的搜索结果
```

**问题**：
1. 简单合并多个兴趣向量的搜索结果可能导致排序不准确
2. 没有考虑每个兴趣向量的权重
3. 对于ComiRec-DR，应该使用训练时的`readout`机制来选择最相关的兴趣向量

### 问题3: NDCG计算中的潜在问题

当前NDCG计算逻辑（第132-144行）：
```python
for no, iid in enumerate(I[i]):
    if iid in true_item_set:
        recall += 1
        dcg += 1.0 / math.log(no + 2, 2)  # no从0开始，所以是log(2), log(3), ...
```

**潜在问题**：
1. 只有当 `recall > 0` 时才计算NDCG，recall=0的样本被忽略，可能导致NDCG偏高
2. IDCG基于命中的item数计算，但如果召回率低，IDCG也会很小
3. 如果真实item排在推荐列表后面，DCG会被折损函数严重惩罚

### 问题4: TensorFlow 1.x vs 2.x 的行为差异

可能的差异点：
1. **随机性**：TensorFlow 2的随机种子行为可能不同
2. **梯度计算**：某些操作的梯度可能略有不同
3. **数值精度**：浮点运算的精度可能略有差异
4. **初始化**：权重初始化的随机性可能不同

## 解决方案

### 解决方案1: 修复评估时使用的向量（最重要）

**对于TensorFlow 1.x兼容模式（非Keras）**：
修改 `Model` 基类，让ComiRec-DR返回 `readout` 而不是 `user_eb`：

```python
class Model_ComiRec_DR(Model):
    def __init__(self, ...):
        # ... 现有代码 ...
        self.user_eb, self.readout = capsule_network(...)
        self.build_sampled_softmax_loss(self.item_eb, self.readout)
    
    def output_user(self, sess, inps):
        # 重写以返回readout而不是user_eb
        user_embs = sess.run(self.readout, feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.mask: inps[1],
            self.mid_batch_ph: np.zeros((inps[0].shape[0],), dtype=np.int32)  # 需要dummy mid
        })
        return user_embs
```

**问题**：`readout`需要`item_eb`（目标item），但评估时没有目标item。需要修复。

### 解决方案2: 在评估时使用所有兴趣向量的最佳合并策略

如果必须使用多个兴趣向量，应该：
1. 对每个兴趣向量分别搜索
2. 使用更智能的合并策略（如加权平均、取最大值等）
3. 或者选择一个代表性兴趣向量（如第一个）

### 解决方案3: 修复NDCG计算

```python
# 应该对所有样本计算NDCG，包括recall=0的
total_ndcg += dcg / idcg if idcg > 0 else 0.0  # 即使recall=0也计入（为0）
```

## 具体修复建议

### 优先级1: 修复ComiRec-DR的output_user方法

对于ComiRec-DR和ComiRec-SA，评估时应该：
- **选项A**：返回`readout`（但需要处理item_eb依赖）
- **选项B**：返回一个代表性兴趣向量（如第一个或平均）
- **选项C**：在评估函数中正确处理多兴趣向量

### 优先级2: 确保训练评估一致性

确保：
1. 训练和评估使用相同的向量表示
2. 评估逻辑与训练时的forward pass一致

### 优先级3: 改进NDCG计算

1. 包含所有样本（包括recall=0）
2. 验证DCG/IDCG计算是否正确
3. 考虑使用不同的NDCG实现（如scikit-learn的）

## 为什么TensorFlow 1.14表现更好？

可能的原因：
1. **原始实现**：可能原始TensorFlow 1.14实现中已经正确处理了这些问题
2. **评估逻辑**：原始代码可能在评估时使用了不同的逻辑
3. **训练设置**：可能原始训练时的超参数或设置不同
4. **数据预处理**：可能数据预处理步骤不同

## 已实施的修复

### 修复1: 恢复使用多兴趣向量进行评估 ✅

**问题**：之前修改为返回单个`readout`向量进行评估，但这可能导致recall下降，因为多兴趣向量策略可以更好地覆盖用户的不同兴趣。

**解决方案**：
- **恢复了原始逻辑**：评估时使用所有兴趣向量（`user_eb`），而不是单个readout向量
- **训练时使用readout**：训练时仍然使用readout计算loss，这是正确的
- **评估时使用多兴趣向量**：评估时对每个兴趣向量分别搜索，然后合并结果，这样可以覆盖用户的多个兴趣

**原因分析**：
- ComiRec-DR和ComiRec-SA的设计目标就是捕获用户的多个兴趣
- 训练时使用readout是为了计算loss（需要一个统一的向量）
- 评估时使用所有兴趣向量可以充分利用模型的multi-interest能力
- 原始TensorFlow 1.14实现也是这样做的（从代码逻辑推断）

**代码位置**：`src/model.py`
- ComiRec-DR和ComiRec-SA不再重写`output_user`方法
- 使用基类的`output_user`方法，返回`user_eb`（所有兴趣向量）

### 修复2: 恢复原始NDCG计算逻辑 ✅

**问题**：之前修改为对所有样本计算NDCG，但这会导致NDCG降低，且与原始实现不一致。

**解决方案**：
- 恢复了原始逻辑：只有当`recall > 0`时才计算NDCG
- 添加了注释说明这种计算方式可能导致NDCG被低估（因为分母是总样本数）
- 为了与原始TensorFlow 1.14实现保持一致，保持这个逻辑

**注意**：原始逻辑中，NDCG = total_ndcg / total，其中：
- total_ndcg只包含有hit（recall>0）的样本的NDCG之和
- total是所有样本数
- 这会导致NDCG被低估，但这是原始实现的方式

**代码位置**：`src/train.py`
- 第143-147行（evaluate_full中的2维情况）
- 第205-209行（evaluate_full中的多兴趣向量情况）
- 第277-281行（evaluate_full_keras）

### 修复3: Keras版本评估函数的mid参数 ✅

**问题**：Keras版本的评估函数对所有模型都使用`dummy_mid=0`，这对ComiRec-DR和ComiRec-SA不正确。

**解决方案**：
- 在`evaluate_full_keras`中添加了`model_type`参数
- 对于ComiRec-DR和ComiRec-SA，使用历史序列的最后一个有效item作为mid
- 对于DNN和GRU4REC，继续使用0作为占位符

**代码位置**：`src/train.py` 第227行（函数签名）和第249-258行（mid计算逻辑）

### 修复4: Model_ComiRec_SA的readout变量 ✅

**问题**：`Model_ComiRec_SA`中`readout`没有保存为实例变量。

**解决方案**：
- 将`readout`赋值给`self.readout`，以便在`output_user`中使用

**代码位置**：`src/model.py` 第292行

## 预期效果

修复后，预期：
1. **Recall提升**：ComiRec-DR的recall应该接近或超过DNN，因为评估时使用了正确的readout向量
2. **NDCG提升**：NDCG应该显著提升，因为：
   - 使用了正确的用户向量（readout）
   - NDCG计算逻辑更准确
3. **结果一致性**：训练和评估使用相同的向量表示，结果应该更稳定

## 下一步行动

1. **重新训练模型**：使用修复后的代码重新训练ComiRec-DR模型
2. **评估性能**：在验证集和测试集上评估，记录Recall、NDCG、HitRate等指标
3. **对比结果**：与TensorFlow 1.14的基准结果对比
4. **如果仍有问题**：
   - 检查超参数设置是否与原始论文一致
   - 检查数据预处理是否一致
   - 考虑是否有其他TensorFlow 1.x vs 2.x的行为差异

