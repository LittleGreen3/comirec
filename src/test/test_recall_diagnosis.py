#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本：分析为什么 ComiRec-DR 的 recall 仍然很低
"""

import sys
import os
import numpy as np
import tensorflow as tf

# 添加路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train import load_item_cate, prepare_data
from data_iterator import DataIterator
from model import KerasModelComiRecDR


def diagnose_model_and_data(model_path, data_path, cate_file, item_count=367983, maxlen=20):
    """诊断模型和数据的问题"""
    print("=" * 60)
    print("ComiRec-DR Recall 低的原因诊断")
    print("=" * 60)
    
    # 1. 检查 item_cate_map
    print("\n[诊断 1] 检查 item_cate_map")
    print("-" * 60)
    item_cate_map = load_item_cate(cate_file)
    print(f"  item_cate_map 大小: {len(item_cate_map)}")
    print(f"  包含 item_id=0: {0 in item_cate_map}")
    print(f"  包含 item_id=1: {1 in item_cate_map}")
    
    # 检查 item_id 范围
    if len(item_cate_map) > 0:
        min_item = min(item_cate_map.keys())
        max_item = max(item_cate_map.keys())
        print(f"  item_id 范围: [{min_item}, {max_item}]")
        print(f"  理论 item_count: {item_count}")
        if max_item < item_count:
            print(f"  ⚠️  警告：item_cate_map 的最大 item_id ({max_item}) < item_count ({item_count})")
            print(f"     这意味着 embedding 层中 item_id > {max_item} 的 item 没有 category 信息")
    
    # 2. 检查数据
    print("\n[诊断 2] 检查训练/验证数据")
    print("-" * 60)
    try:
        valid_data = DataIterator(data_path, batch_size=10, maxlen=maxlen, train_flag=1)
        sample_count = 0
        item_id_set = set()
        hist_lengths = []
        invalid_items = []
        
        for src, tgt in valid_data:
            if sample_count >= 100:  # 只检查前 100 个样本
                break
            
            nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
            
            for i in range(len(item_id)):
                # 收集 item_id
                if isinstance(item_id[i], list):
                    item_id_set.update(item_id[i])
                else:
                    item_id_set.add(item_id[i])
                
                # 检查历史序列长度
                hist_seq = hist_item[i]
                valid_hist = [x for x in hist_seq if x != 0]
                hist_lengths.append(len(valid_hist))
                
                # 检查是否有无效 item
                if isinstance(item_id[i], list):
                    for itm in item_id[i]:
                        if itm == 0 or itm not in item_cate_map:
                            invalid_items.append(itm)
                else:
                    if item_id[i] == 0 or item_id[i] not in item_cate_map:
                        invalid_items.append(item_id[i])
            
            sample_count += len(item_id)
        
        print(f"  检查了 {sample_count} 个样本")
        print(f"  唯一 item_id 数量: {len(item_id_set)}")
        print(f"  平均历史序列长度: {np.mean(hist_lengths):.2f}")
        print(f"  历史序列长度范围: [{min(hist_lengths)}, {max(hist_lengths)}]")
        
        if len(invalid_items) > 0:
            print(f"  ⚠️  警告：发现 {len(invalid_items)} 个无效 item_id")
            print(f"     无效 item_id 示例: {set(invalid_items[:10])}")
        
        # 检查 item_id 范围
        if len(item_id_set) > 0:
            min_item_data = min(item_id_set)
            max_item_data = max(item_id_set)
            print(f"  数据中 item_id 范围: [{min_item_data}, {max_item_data}]")
            
            if max_item_data >= item_count:
                print(f"  ⚠️  警告：数据中有 item_id ({max_item_data}) >= item_count ({item_count})")
                print(f"     这会导致 embedding 索引越界问题！")
        
    except Exception as e:
        print(f"  ✗ 无法读取数据: {e}")
    
    # 3. 检查模型配置
    print("\n[诊断 3] 检查模型配置")
    print("-" * 60)
    print(f"  item_count: {item_count}")
    print(f"  maxlen: {maxlen}")
    print(f"  模型类型: ComiRec-DR")
    
    # 4. 诊断建议
    print("\n[诊断 4] 可能的改进建议")
    print("-" * 60)
    print("  1. 确保训练和评估时 mid 的使用方式一致")
    print("     - 训练时：使用真实的 item_id 作为 mid")
    print("     - 评估时：使用历史序列最后一个有效 item 作为 mid")
    print("")
    print("  2. 检查模型是否充分训练")
    print("     - loss 应该持续下降（当前 6.4-6.6 仍然很高）")
    print("     - 可能需要更多训练迭代")
    print("")
    print("  3. 检查学习率和优化器设置")
    print("     - 当前 learning_rate=0.001 可能偏大或偏小")
    print("     - 可以尝试调整学习率")
    print("")
    print("  4. 检查采样负样本数量")
    print("     - 当前 neg_num=10，可能需要调整")
    print("")
    print("  5. 验证模型架构实现是否正确")
    print("     - 确保 capsule 网络的 routing 机制正确")
    print("     - 确保 attention 机制正确计算")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='诊断 ComiRec-DR recall 低的问题')
    parser.add_argument('--data_path', type=str, default='./data/book_data/book_valid.txt',
                        help='验证数据路径')
    parser.add_argument('--cate_file', type=str, default='./data/book_data/book_item_cate.txt',
                        help='类别文件路径')
    parser.add_argument('--item_count', type=int, default=367983,
                        help='item 总数')
    parser.add_argument('--maxlen', type=int, default=20,
                        help='最大序列长度')
    
    args = parser.parse_args()
    
    diagnose_model_and_data(
        model_path=None,  # 暂时不需要加载模型
        data_path=args.data_path,
        cate_file=args.cate_file,
        item_count=args.item_count,
        maxlen=args.maxlen
    )


if __name__ == '__main__':
    main()

