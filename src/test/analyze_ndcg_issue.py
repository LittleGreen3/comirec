#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析为什么 NDCG 很低
检查 NDCG 计算逻辑和可能的问题
"""

import math
import numpy as np


def calculate_ndcg_example():
    """演示 NDCG 计算"""
    print("=" * 60)
    print("NDCG 计算示例分析")
    print("=" * 60)
    
    # 示例1：理想情况（所有真实item都在最前面）
    print("\n[示例1] 理想情况")
    print("-" * 60)
    pred_items = [1, 2, 3, 4, 5]  # 推荐的物品
    true_items = [1, 2, 3]  # 真实标签
    true_set = set(true_items)
    
    dcg = 0.0
    for pos, item in enumerate(pred_items):
        if item in true_set:
            dcg += 1.0 / math.log(pos + 2, 2)
            print(f"  位置 {pos+1}: item {item} ✓, DCG贡献 = {1.0 / math.log(pos + 2, 2):.4f}")
    
    idcg = sum(1.0 / math.log(i + 2, 2) for i in range(len(true_items)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    print(f"  DCG: {dcg:.4f}")
    print(f"  IDCG: {idcg:.4f}")
    print(f"  NDCG: {ndcg:.4f} ✓")
    
    # 示例2：真实item都在后面
    print("\n[示例2] 真实item都在后面（常见问题）")
    print("-" * 60)
    pred_items = [100, 200, 300, 1, 2, 3]  # 真实item在后面
    true_items = [1, 2, 3]
    true_set = set(true_items)
    
    dcg = 0.0
    hit_positions = []
    for pos, item in enumerate(pred_items):
        if item in true_set:
            dcg += 1.0 / math.log(pos + 2, 2)
            hit_positions.append(pos + 1)
            print(f"  位置 {pos+1}: item {item} ✓, DCG贡献 = {1.0 / math.log(pos + 2, 2):.4f}")
    
    idcg = sum(1.0 / math.log(i + 2, 2) for i in range(len(true_items)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    print(f"  命中位置: {hit_positions}")
    print(f"  DCG: {dcg:.4f}")
    print(f"  IDCG: {idcg:.4f}")
    print(f"  NDCG: {ndcg:.4f} ⚠️ 很低！")
    print(f"  原因：真实item排在后面，DCG被折损函数惩罚")
    
    # 示例3：只有部分命中
    print("\n[示例3] 只有部分命中（recall低）")
    print("-" * 60)
    pred_items = [1, 100, 200, 300, 400, 500]  # 只有1个命中
    true_items = [1, 2, 3, 4, 5]  # 5个真实标签
    true_set = set(true_items)
    
    dcg = 0.0
    recall = 0
    for pos, item in enumerate(pred_items):
        if item in true_set:
            recall += 1
            dcg += 1.0 / math.log(pos + 2, 2)
            print(f"  位置 {pos+1}: item {item} ✓, DCG贡献 = {1.0 / math.log(pos + 2, 2):.4f}")
    
    idcg = sum(1.0 / math.log(i + 2, 2) for i in range(recall)) if recall > 0 else 0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    recall_rate = recall / len(true_items)
    
    print(f"  Recall: {recall}/{len(true_items)} = {recall_rate:.4f}")
    print(f"  DCG: {dcg:.4f}")
    print(f"  IDCG: {idcg:.4f} (只基于命中的{recall}个)")
    print(f"  NDCG: {ndcg:.4f} ⚠️ 虽然item在位置1，但recall低导致总体NDCG低")


def analyze_current_ndcg_issue():
    """分析当前NDCG低的问题"""
    print("\n" + "=" * 60)
    print("当前 NDCG 低的原因分析")
    print("=" * 60)
    
    print("\n[问题1] Recall低导致NDCG低")
    print("-" * 60)
    print("  当前 recall ≈ 0.02，意味着：")
    print("  - 100个真实item中，只有2个被预测到")
    print("  - 即使这2个item排在前面，DCG仍然很低")
    print("  - IDCG基于命中的item数计算，如果只命中2个，IDCG也很小")
    print("  ✓ 解决方法：提高recall（这是根本问题）")
    
    print("\n[问题2] 排序质量差")
    print("-" * 60)
    print("  即使命中了真实item，但如果它们排在推荐列表的后面：")
    print("  - 位置1: DCG = 1.0 / log2(3) = 0.63")
    print("  - 位置10: DCG = 1.0 / log2(12) = 0.29")
    print("  - 位置50: DCG = 1.0 / log2(52) = 0.19")
    print("  ⚠️  位置越靠后，DCG贡献越小")
    print("  ✓ 解决方法：提高模型排序质量，让真实item排在前面")
    
    print("\n[问题3] NDCG计算逻辑检查")
    print("-" * 60)
    print("  当前代码逻辑：")
    print("  - 只有当 recall > 0 时才计算 NDCG")
    print("  - NDCG = DCG / IDCG")
    print("  - IDCG基于命中的item数计算（这是正确的）")
    print("  ⚠️  问题：如果recall=0，NDCG不计入统计，可能导致NDCG偏高")
    print("  ✓ 建议：所有样本都应该计入NDCG统计（包括recall=0的）")
    
    print("\n[问题4] Faiss搜索的排序问题")
    print("-" * 60)
    print("  Faiss返回的结果是按相似度（内积）排序的")
    print("  如果模型学习不好，相似度计算可能不准确：")
    print("  - user_vec和item_emb的相似度可能不反映真实的用户偏好")
    print("  - 导致真实item排在后面")
    print("  ✓ 解决方法：改进模型训练，提高embedding质量")


def suggest_improvements():
    """建议改进措施"""
    print("\n" + "=" * 60)
    print("改进建议")
    print("=" * 60)
    
    print("\n[建议1] 提高Recall（最关键）")
    print("-" * 60)
    print("  1. 增加负样本数量（已修改：neg_num 10 → 50）")
    print("  2. 降低学习率（已修改：lr 0.001 → 0.0005）")
    print("  3. 增加训练时间")
    print("  4. 检查模型架构实现是否正确")
    
    print("\n[建议2] 改进排序质量")
    print("-" * 60)
    print("  1. 确保user_vec和item_emb在同一空间（当前应该是）")
    print("  2. 使用更好的损失函数（考虑排序的pairwise loss）")
    print("  3. 增加embedding维度（可能需要实验）")
    
    print("\n[建议3] 修复NDCG计算")
    print("-" * 60)
    print("  当前代码只对recall>0的样本计算NDCG")
    print("  应该改为：对所有样本计算NDCG（recall=0时NDCG=0）")
    print("  这样更公平地反映模型整体性能")


def main():
    calculate_ndcg_example()
    analyze_current_ndcg_issue()
    suggest_improvements()
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)
    print("\n关键发现：")
    print("1. NDCG低主要是因为Recall低（0.02）")
    print("2. 即使命中的item排在前面，因为recall低，总体NDCG仍然很低")
    print("3. 如果命中的item排在后面，NDCG会更低")
    print("4. 提高recall是最重要的（已通过增加负样本和降低学习率优化）")


if __name__ == '__main__':
    main()

