# 修复总结

## 已修复的问题

### 1. KeyError: 0 错误 ✅
**问题**：`compute_diversity` 函数访问 `item_cate_map[0]` 时抛出 KeyError，因为 item_id=0 不在 item_cate_map 中。

**修复**：
- 在所有评估函数中，在计算 recall/ndcg 和 diversity 之前，过滤掉无效的 item_id
- 过滤条件：
  - 排除 item_id <= 0（0 是 padding 值）
  - 排除不在 `item_cate_map` 中的 item（如果提供了 item_cate_map）
  - 排除超出 embedding 层范围的 item

### 2. ComiRec-DR 模型训练和评估不一致 ✅
**问题**：
- 训练时使用 `dummy_mid=0`（无效 item）作为 mid，导致模型无法正确学习
- 评估时也使用 `dummy_mid=0`，导致性能下降

**修复**：
- **训练时**：对于需要 mid 的模型（ComiRec-DR, ComiRec-SA, MIND），使用真实的 `item_id` 作为 mid
- **评估时**：使用历史序列中最后一个有效 item 作为 mid（确保在有效范围内）

### 3. 诊断脚本错误 ✅
**问题**：诊断脚本中的数据读取逻辑错误（`tuple index out of range`）

**修复**：使用 `prepare_data` 函数正确解析数据

### 4. Item ID 边界处理 ✅
**问题**：item_cate_map 范围是 [1, 367982]，但 item_count=367983，可能存在边界 item 未处理

**修复**：
- 在过滤逻辑中严格检查 item_id 范围
- 在评估时选择 mid 时，确保 item_id 在有效范围内（< len(item_embs)）

## 代码改进

### 训练代码 (`src/train.py`)
1. **`evaluate_full_keras`** 函数：
   - 改进了 mid 选择策略，确保不会使用超出范围的 item_id
   - 改进了无效 item 过滤逻辑

2. **`evaluate_full`** 函数：
   - 同样改进了无效 item 过滤逻辑

3. **训练循环**：
   - ComiRec 模型使用真实的 item_id 作为 mid

### 测试脚本
1. **`test_diversity_keyerror.py`**：验证 KeyError 修复
2. **`test_recall_diagnosis.py`**：诊断 recall 低的原因（已修复数据读取错误）

## Recall 低的原因分析

根据诊断结果和训练日志：

### 主要问题
1. **模型训练不充分**：
   - Loss 从 6.65 降到 6.43，仍在下降，但数值偏高
   - 需要更多训练迭代

2. **模型复杂度**：
   - ComiRec-DR 使用多兴趣建模，比 DNN 更复杂
   - 需要更多数据和时间来训练

3. **超参数设置**：
   - `learning_rate=0.001` 可能需要调整
   - `neg_num=10` 可能需要调整
   - `num_interest=4` 需要验证是否合适

### 建议
1. **继续训练**：观察 loss 和 recall 是否持续改善
2. **调整学习率**：尝试 0.0005 或使用学习率衰减
3. **增加训练迭代**：ComiRec-DR 通常需要更多迭代
4. **调整超参数**：尝试不同的 `num_interest` 和 `neg_num`

## 验证

运行测试脚本验证修复：
```bash
# 测试 KeyError 修复
python src/test/test_diversity_keyerror.py

# 诊断 recall 问题
python src/test/test_recall_diagnosis.py \
    --data_path ./data/book_data/book_valid.txt \
    --cate_file ./data/book_data/book_item_cate.txt \
    --item_count 367983 \
    --maxlen 20
```

## 预期效果

修复后：
- ✅ 不会再出现 KeyError
- ✅ 模型能正确学习（使用有效的 mid）
- ✅ 评估时使用合理的 mid
- ✅ Recall 应该会随着训练逐步提升（需要更多训练时间）

## 注意事项

1. **Item ID 范围**：确保所有 item_id 都在 [1, item_count-1] 范围内
2. **Item Cate Map**：item_cate_map 只包含有效的 item（不包含 0）
3. **训练时间**：ComiRec-DR 需要比 DNN 更多的训练时间来收敛

