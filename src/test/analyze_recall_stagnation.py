#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析为什么 recall 在 0.02 附近徘徊
检查训练和评估逻辑的问题
"""

import sys
import os
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train import prepare_data, load_item_cate
from data_iterator import DataIterator
from model import KerasModelComiRecDR


def analyze_training_problem(item_count=367983, batch_size=128, maxlen=20):
    """分析训练问题"""
    print("=" * 60)
    print("分析 Recall 停滞在 0.02 的原因")
    print("=" * 60)
    
    print("\n[问题 1] 检查 sampled_softmax_loss 的配置")
    print("-" * 60)
    neg_num = 10
    num_sampled = neg_num * batch_size  # 10 * 128 = 1280
    print(f"  neg_num: {neg_num}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_sampled (负样本数): {num_sampled}")
    print(f"  item_count: {item_count}")
    print(f"  num_sampled / item_count = {num_sampled / item_count * 100:.4f}%")
    print(f"  ⚠️  负样本数量相对总 item 数太少，可能导致模型学习不充分")
    
    print("\n[问题 2] 检查学习率和优化器")
    print("-" * 60)
    lr = 0.001
    print(f"  learning_rate: {lr}")
    print(f"  optimizer: Adam")
    print(f"  ⚠️  对于复杂模型（ComiRec-DR），lr=0.001 可能偏大，导致训练不稳定")
    
    print("\n[问题 3] 检查模型复杂度")
    print("-" * 60)
    num_interest = 4
    embedding_dim = 64
    hidden_size = 64
    print(f"  num_interest: {num_interest}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  maxlen: {maxlen}")
    print(f"  ⚠️  ComiRec-DR 是多兴趣模型，需要更多训练时间和数据")
    
    print("\n[问题 4] 检查评估逻辑")
    print("-" * 60)
    print(f"  评估时使用历史序列最后一个 item 作为 mid")
    print(f"  训练时使用真实的 item_id 作为 mid")
    print(f"  ⚠️  训练和评估时 mid 的使用方式不同，可能导致不一致")
    
    print("\n[问题 5] 检查 loss 和 recall 的关系")
    print("-" * 60)
    print(f"  Loss: 6.77 -> 6.53 (在下降)")
    print(f"  Recall: 0.019-0.020 (几乎没有提升)")
    print(f"  ⚠️  Loss 下降但 recall 不提升，可能的原因：")
    print(f"      1. 模型过拟合到负采样分布，但没有学到真实的用户偏好")
    print(f"      2. sampled_softmax_loss 的负样本数量太少")
    print(f"      3. 学习率可能不合适")
    print(f"      4. 模型架构实现可能有问题")
    
    print("\n[建议 1] 增加负样本数量")
    print("-" * 60)
    print(f"  当前: neg_num = {neg_num}")
    print(f"  建议: neg_num = 50-100 (增加负样本数，提高学习质量)")
    
    print("\n[建议 2] 调整学习率")
    print("-" * 60)
    print(f"  当前: lr = {lr}")
    print(f"  建议: lr = 0.0005 或使用学习率衰减")
    print(f"        或者使用 warmup 策略")
    
    print("\n[建议 3] 检查模型输出")
    print("-" * 60)
    print(f"  检查 user_vec 的范数是否正常")
    print(f"  检查 embedding 是否在合理范围内")
    print(f"  检查梯度是否正常流动")
    
    print("\n[建议 4] 尝试不同的评估策略")
    print("-" * 60)
    print(f"  当前: 使用历史最后一个 item 作为 mid")
    print(f"  建议: 尝试使用多个候选 mid 并取平均")
    print(f"        或者使用一个固定的代表性 item")
    
    print("\n[建议 5] 对比 DNN baseline")
    print("-" * 60)
    print(f"  如果 DNN 的 recall 正常，说明数据没问题")
    print(f"  如果 DNN 的 recall 也很低，可能是数据或评估逻辑的问题")
    
    print("\n" + "=" * 60)


def check_model_output_shape():
    """检查模型输出形状是否正确"""
    print("\n[诊断] 检查模型输出形状")
    print("-" * 60)
    
    try:
        # 创建模型
        item_count = 367983
        maxlen = 20
        model = KerasModelComiRecDR(
            n_mid=item_count,
            embedding_dim=64,
            hidden_size=64,
            num_interest=4,
            seq_len=maxlen
        )
        
        # 测试输入
        batch_size = 2
        dummy_mid = tf.zeros((batch_size,), dtype=tf.int32)
        hist_item = tf.zeros((batch_size, maxlen), dtype=tf.int32)
        hist_mask = tf.ones((batch_size, maxlen), dtype=tf.float32)
        
        # 前向传播
        user_vec, item_vec = model([dummy_mid, hist_item, hist_mask], training=False)
        
        print(f"  user_vec shape: {user_vec.shape}")
        print(f"  item_vec shape: {item_vec.shape}")
        print(f"  ✓ 输出形状正确")
        
        # 检查输出范围
        user_norm = tf.norm(user_vec, axis=1)
        print(f"  user_vec 范数: {user_norm.numpy()}")
        
        if tf.reduce_max(tf.abs(user_vec)) > 100:
            print(f"  ⚠️  警告: user_vec 的值可能过大")
        
    except Exception as e:
        print(f"  ✗ 模型测试失败: {e}")


def main():
    analyze_training_problem()
    check_model_output_shape()
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


if __name__ == '__main__':
    main()

