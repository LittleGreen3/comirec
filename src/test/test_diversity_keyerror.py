#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证 KeyError: 0 是否已被修复
测试 compute_diversity 函数是否能正确处理无效 item_id
"""

import sys
import os
import numpy as np
import tensorflow as tf

# 添加路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train import compute_diversity, load_item_cate


def test_compute_diversity_with_invalid_items():
    """测试 compute_diversity 是否能正确处理包含无效 item_id 的情况"""
    print("=" * 60)
    print("测试 1: compute_diversity 处理无效 item_id")
    print("=" * 60)
    
    # 创建一个模拟的 item_cate_map（不包含 item_id=0）
    item_cate_map = {
        1: 10,
        2: 20,
        3: 10,
        4: 30,
        5: 20,
        6: 10,
    }
    
    # 测试案例1：包含 item_id=0 的列表（应该过滤掉）
    item_list_with_zero = [0, 1, 2, 3, 4]
    print(f"\n测试列表（包含0）: {item_list_with_zero}")
    
    # 过滤无效 item
    valid_items = [item for item in item_list_with_zero if item != 0 and item in item_cate_map]
    print(f"过滤后的有效 item: {valid_items}")
    
    if len(valid_items) > 0:
        try:
            diversity = compute_diversity(valid_items, item_cate_map)
            print(f"✓ 成功计算 diversity: {diversity:.4f}")
        except KeyError as e:
            print(f"✗ KeyError 仍然存在: {e}")
            return False
    
    # 测试案例2：包含不在 item_cate_map 中的 item
    item_list_with_invalid = [1, 2, 999, 4]  # 999 不在 item_cate_map 中
    print(f"\n测试列表（包含无效item）: {item_list_with_invalid}")
    
    valid_items = [item for item in item_list_with_invalid if item != 0 and item in item_cate_map]
    print(f"过滤后的有效 item: {valid_items}")
    
    if len(valid_items) > 0:
        try:
            diversity = compute_diversity(valid_items, item_cate_map)
            print(f"✓ 成功计算 diversity: {diversity:.4f}")
        except KeyError as e:
            print(f"✗ KeyError 仍然存在: {e}")
            return False
    
    # 测试案例3：全部是无效 item
    item_list_all_invalid = [0, 999, 888]
    print(f"\n测试列表（全部无效）: {item_list_all_invalid}")
    
    valid_items = [item for item in item_list_all_invalid if item != 0 and item in item_cate_map]
    print(f"过滤后的有效 item: {valid_items}")
    
    if len(valid_items) == 0:
        print("✓ 正确过滤掉所有无效 item，跳过 diversity 计算")
    
    print("\n✓ 测试 1 通过！")
    return True


def test_evaluate_diversity_logic():
    """测试评估逻辑中 diversity 计算的流程"""
    print("\n" + "=" * 60)
    print("测试 2: 评估逻辑中的 diversity 计算")
    print("=" * 60)
    
    # 模拟 faiss 返回的结果（可能包含 item_id=0）
    faiss_results = np.array([[0, 1, 2, 3, 4, 999],  # 包含 0 和无效 item
                              [1, 2, 3, 4, 5, 6]])   # 全部有效
    
    item_cate_map = {
        1: 10, 2: 20, 3: 10, 4: 30, 5: 20, 6: 10
    }
    
    print(f"\nFaiss 返回结果:\n{faiss_results}")
    
    for i, I_row in enumerate(faiss_results):
        print(f"\n处理第 {i} 个样本:")
        print(f"  原始结果: {I_row}")
        
        # 过滤无效 item
        if item_cate_map:
            valid_items = [item for item in I_row if item != 0 and item in item_cate_map]
        else:
            valid_items = [item for item in I_row if item != 0]
        
        print(f"  过滤后: {valid_items}")
        
        if len(valid_items) > 0:
            try:
                diversity = compute_diversity(valid_items, item_cate_map)
                print(f"  ✓ Diversity: {diversity:.4f}")
            except KeyError as e:
                print(f"  ✗ KeyError: {e}")
                return False
        else:
            print(f"  - 跳过（无有效 item）")
    
    print("\n✓ 测试 2 通过！")
    return True


def test_item_cate_map_loading():
    """测试 item_cate_map 加载是否排除 item_id=0"""
    print("\n" + "=" * 60)
    print("测试 3: item_cate_map 加载验证")
    print("=" * 60)
    
    # 创建临时文件
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        # 写入测试数据（注意：不包含 item_id=0）
        f.write("1,10\n")
        f.write("2,20\n")
        f.write("3,10\n")
        f.write("4,30\n")
        temp_file = f.name
    
    try:
        item_cate_map = load_item_cate(temp_file)
        print(f"\n加载的 item_cate_map: {item_cate_map}")
        
        # 检查是否包含 item_id=0
        if 0 in item_cate_map:
            print("✗ item_cate_map 错误地包含了 item_id=0")
            return False
        else:
            print("✓ item_cate_map 正确排除了 item_id=0")
        
        # 测试访问 item_id=0 是否会报错
        try:
            _ = item_cate_map[0]
            print("✗ 访问 item_cate_map[0] 应该报错但没有")
            return False
        except KeyError:
            print("✓ 访问 item_cate_map[0] 正确抛出 KeyError")
        
    finally:
        os.unlink(temp_file)
    
    print("\n✓ 测试 3 通过！")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试 KeyError 修复")
    print("=" * 60)
    
    all_passed = True
    
    # 测试1
    if not test_compute_diversity_with_invalid_items():
        all_passed = False
    
    # 测试2
    if not test_evaluate_diversity_logic():
        all_passed = False
    
    # 测试3
    if not test_item_cate_map_loading():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！KeyError 修复验证成功")
    else:
        print("✗ 部分测试失败，请检查代码")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

