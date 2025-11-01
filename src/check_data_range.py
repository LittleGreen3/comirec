#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断工具：检查训练数据中的 item_id 范围是否与 item_count 匹配

用法：
    python check_data_range.py --dataset book
    python check_data_range.py --dataset taobao
"""

import argparse
import sys
import os

# 添加路径以便导入模块
sys.path.insert(0, os.path.dirname(__file__))

from data_iterator import DataIterator


def check_data_range(data_file, item_count, max_samples=10000):
    """
    检查数据文件中的 item_id 范围
    
    Args:
        data_file: 数据文件路径
        item_count: 模型期望的 item_count
        max_samples: 最多检查的样本数
    """
    print("=" * 80)
    print(f"检查数据文件: {data_file}")
    print(f"期望 item_count: {item_count}")
    print("=" * 80)
    print()
    
    if not os.path.exists(data_file):
        print(f"❌ 错误：文件不存在: {data_file}")
        return
    
    # 读取数据并收集所有 item_id
    all_item_ids = set()
    invalid_item_ids = set()
    invalid_samples = []
    sample_count = 0
    
    try:
        iterator = DataIterator(data_file, batch_size=128, maxlen=100, train_flag=1)
        print("📊 正在读取数据...")
        
        for src, tgt in iterator:
            user_id_list, item_id_list = src
            hist_item_list, hist_mask_list = tgt
            
            for i in range(len(item_id_list)):
                sample_count += 1
                if sample_count > max_samples:
                    break
                
                # 检查目标 item_id
                target_item = item_id_list[i]
                if isinstance(target_item, list):
                    # 测试集（多个目标 item）
                    for item_id in target_item:
                        all_item_ids.add(item_id)
                        if item_id >= item_count or item_id <= 0:
                            invalid_item_ids.add(item_id)
                            invalid_samples.append((sample_count, item_id, 'target'))
                else:
                    # 训练集（单个目标 item）
                    all_item_ids.add(target_item)
                    if target_item >= item_count or target_item <= 0:
                        invalid_item_ids.add(target_item)
                        invalid_samples.append((sample_count, target_item, 'target'))
                
                # 检查历史序列中的 item_id
                hist_items = hist_item_list[i]
                for item_id in hist_items:
                    if item_id > 0:  # 0 是 padding，忽略
                        all_item_ids.add(item_id)
                        if item_id >= item_count:
                            invalid_item_ids.add(item_id)
                            invalid_samples.append((sample_count, item_id, 'history'))
            
            if sample_count > max_samples:
                break
    except Exception as e:
        print(f"❌ 读取数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 分析结果
    print()
    print("=" * 80)
    print("检查结果")
    print("=" * 80)
    print(f"📊 检查了 {sample_count} 个样本")
    print(f"📊 发现 {len(all_item_ids)} 个唯一的 item_id")
    
    if len(all_item_ids) > 0:
        min_item_id = min(all_item_ids)
        max_item_id = max(all_item_ids)
        print(f"📊 item_id 范围: [{min_item_id}, {max_item_id}]")
        print(f"📊 期望范围: [1, {item_count})")
        print()
    
    if len(invalid_item_ids) > 0:
        print(f"⚠️  发现 {len(invalid_item_ids)} 个无效的 item_id (超出范围 [1, {item_count}))")
        print()
        print("无效 item_id 列表（前20个）:")
        sorted_invalid = sorted(list(invalid_item_ids))
        for i, item_id in enumerate(sorted_invalid[:20]):
            print(f"  {i+1}. item_id={item_id}")
        if len(sorted_invalid) > 20:
            print(f"  ... 还有 {len(sorted_invalid) - 20} 个")
        print()
        print("无效样本示例（前10个）:")
        for i, (sample_idx, item_id, item_type) in enumerate(invalid_samples[:10]):
            print(f"  {i+1}. 样本 #{sample_idx}: {item_type} item_id={item_id}")
        print()
        print("=" * 80)
        print("⚠️  建议修复方案")
        print("=" * 80)
        print()
        print("方案 1: 重新预处理数据（推荐）")
        print("   - 检查预处理脚本是否正确过滤了所有 item")
        print("   - 确保 item_map 包含所有训练/验证/测试数据中出现的 item")
        print("   - 重新运行预处理脚本生成数据")
        print()
        print("方案 2: 更新 item_count")
        print(f"   - 当前 item_count={item_count}")
        print(f"   - 实际最大 item_id={max_item_id}")
        print(f"   - 建议 item_count >= {max_item_id + 1}")
        print(f"   - 修改 src/train.py 第601行（book）或第595行（taobao）")
        print(f"   - 将 item_count 改为: {max_item_id + 1}")
        print()
        print("方案 3: 使用数据验证（临时方案）")
        print("   - 代码中已添加自动过滤功能")
        print("   - 无效样本会被自动跳过")
        print("   - 但建议从根本上修复数据问题")
    else:
        print("✅ 所有 item_id 都在有效范围内 [1, {})".format(item_count))
        print()
        print("数据范围检查通过！")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='检查训练数据中的 item_id 范围')
    parser.add_argument('--dataset', type=str, default='book', choices=['book', 'taobao'],
                        help='数据集名称')
    parser.add_argument('--data_file', type=str, default=None,
                        help='数据文件路径（如果不指定，将使用默认路径）')
    parser.add_argument('--item_count', type=int, default=None,
                        help='item_count（如果不指定，将使用默认值）')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='最多检查的样本数（默认：10000）')
    
    args = parser.parse_args()
    
    # 确定默认路径和 item_count
    if args.dataset == 'taobao':
        default_path = './data/taobao_data/taobao_train.txt'
        default_item_count = 1708531
    else:  # book
        default_path = './data/book_data/book_train.txt'
        default_item_count = 367983
    
    data_file = args.data_file if args.data_file else default_path
    item_count = args.item_count if args.item_count else default_item_count
    
    check_data_range(data_file, item_count, args.max_samples)


if __name__ == '__main__':
    main()

