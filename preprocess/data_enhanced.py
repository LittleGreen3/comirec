"""
增强版数据预处理脚本
支持自定义数据分割比例和数据采样
"""
import os
import sys
import json
import random
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='数据预处理工具')
    parser.add_argument('dataset', type=str, choices=['book', 'taobao'], 
                       help='数据集名称')
    parser.add_argument('--filter_size', type=int, default=5,
                       help='最小交互数过滤（默认：5）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例（默认：0.8）')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                       help='验证集比例（默认：0.1）')
    parser.add_argument('--test_ratio', type=float, default=None,
                       help='测试集比例（默认：自动计算 = 1 - train_ratio - valid_ratio）')
    parser.add_argument('--max_users', type=int, default=None,
                       help='最大用户数（用于数据采样，默认：不限制）')
    parser.add_argument('--seed', type=int, default=1230,
                       help='随机种子（默认：1230）')
    parser.add_argument('--stats', action='store_true',
                       help='输出详细统计信息')
    
    args = parser.parse_args()
    
    # 验证比例
    if args.test_ratio is None:
        args.test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"⚠️  警告：比例总和 = {total_ratio:.3f}，不等于 1.0")
        print(f"   自动归一化：train={args.train_ratio/total_ratio:.3f}, "
              f"valid={args.valid_ratio/total_ratio:.3f}, "
              f"test={args.test_ratio/total_ratio:.3f}")
        args.train_ratio /= total_ratio
        args.valid_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    return args

def read_from_amazon(source):
    users = defaultdict(list)
    item_count = defaultdict(int)
    
    print(f"正在读取 Amazon 数据：{source}")
    line_count = 0
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line.strip())
            uid = r['user_id']
            iid = r['asin']
            item_count[iid] += 1
            ts = float(r['timestamp'])
            users[uid].append((iid, ts))
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  已读取 {line_count} 行...")
    
    print(f"✅ 读取完成：{len(users)} 用户，{len(item_count)} 物品")
    return users, item_count

def read_from_taobao(source):
    users = defaultdict(list)
    item_count = defaultdict(int)
    
    print(f"正在读取 Taobao 数据：{source}")
    line_count = 0
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            conts = line.strip().split(',')
            uid = int(conts[0])
            iid = int(conts[1])
            if conts[3] != 'pv':
                continue
            item_count[iid] += 1
            ts = int(conts[4])
            users[uid].append((iid, ts))
            line_count += 1
            if line_count % 100000 == 0:
                print(f"  已读取 {line_count} 行...")
    
    print(f"✅ 读取完成：{len(users)} 用户，{len(item_count)} 物品")
    return users, item_count

def filter_items(items, item_count, filter_size):
    """过滤交互数少于 filter_size 的物品"""
    items = list(items)
    items.sort(key=lambda x: x[1], reverse=True)
    
    item_total = 0
    for index, (iid, num) in enumerate(items):
        if num >= filter_size:
            item_total = index + 1
        else:
            break
    
    item_map = dict(zip([items[i][0] for i in range(item_total)], 
                       list(range(1, item_total+1))))
    
    print(f"📊 过滤后：{item_total} 个物品（最小交互数 >= {filter_size}）")
    return item_map

def filter_users(users, item_map, filter_size):
    """过滤交互数少于 filter_size 的用户"""
    user_ids = list(users.keys())
    filter_user_ids = []
    
    for user in user_ids:
        item_list = users[user]
        index = 0
        for item, timestamp in item_list:
            if item in item_map:
                index += 1
        if index >= filter_size:
            filter_user_ids.append(user)
    
    print(f"📊 过滤后：{len(filter_user_ids)} 个用户（最小交互数 >= {filter_size}）")
    return filter_user_ids

def sample_users(user_ids, max_users, seed):
    """采样用户（如果需要）"""
    if max_users is None or len(user_ids) <= max_users:
        return user_ids
    
    print(f"🔄 采样用户：{len(user_ids)} → {max_users}")
    random.seed(seed)
    return random.sample(user_ids, max_users)

def split_users(user_ids, train_ratio, valid_ratio, test_ratio, seed):
    """按比例分割用户"""
    random.seed(seed)
    random.shuffle(user_ids)
    
    num_users = len(user_ids)
    split_1 = int(num_users * train_ratio)
    split_2 = int(num_users * (train_ratio + valid_ratio))
    
    train_users = user_ids[:split_1]
    valid_users = user_ids[split_1:split_2]
    test_users = user_ids[split_2:]
    
    print(f"\n📐 数据分割：")
    print(f"   训练集：{len(train_users)} 用户 ({train_ratio*100:.1f}%)")
    print(f"   验证集：{len(valid_users)} 用户 ({valid_ratio*100:.1f}%)")
    print(f"   测试集：{len(test_users)} 用户 ({test_ratio*100:.1f}%)")
    
    return train_users, valid_users, test_users

def export_map(name, map_dict):
    """导出映射文件"""
    with open(name, 'w', encoding='utf-8') as f:
        for key, value in map_dict.items():
            f.write('%s,%d\n' % (key, value))

def export_data(name, user_list, users, item_map, user_map):
    """导出数据文件"""
    total_data = 0
    with open(name, 'w', encoding='utf-8') as f:
        for user in user_list:
            if user not in user_map:
                continue
            item_list = users[user]
            item_list.sort(key=lambda x: x[1])
            index = 0
            for item, timestamp in item_list:
                if item in item_map:
                    f.write('%d,%d,%d\n' % (user_map[user], item_map[item], index))
                    index += 1
                    total_data += 1
    return total_data

def print_statistics(users, item_map, user_map, train_users, valid_users, test_users):
    """输出详细统计信息"""
    print("\n" + "="*60)
    print("📊 详细统计信息")
    print("="*60)
    
    # 用户统计
    def get_user_stats(user_list):
        total_interactions = 0
        interaction_counts = []
        for user in user_list:
            if user not in users:
                continue
            count = sum(1 for item, _ in users[user] if item in item_map)
            interaction_counts.append(count)
            total_interactions += count
        return {
            'count': len(user_list),
            'total_interactions': total_interactions,
            'avg_interactions': total_interactions / len(user_list) if user_list else 0,
            'min_interactions': min(interaction_counts) if interaction_counts else 0,
            'max_interactions': max(interaction_counts) if interaction_counts else 0,
        }
    
    train_stats = get_user_stats(train_users)
    valid_stats = get_user_stats(valid_users)
    test_stats = get_user_stats(test_users)
    
    print(f"\n训练集：")
    print(f"  用户数：{train_stats['count']:,}")
    print(f"  交互总数：{train_stats['total_interactions']:,}")
    print(f"  平均交互/用户：{train_stats['avg_interactions']:.1f}")
    print(f"  交互范围：{train_stats['min_interactions']} - {train_stats['max_interactions']}")
    
    print(f"\n验证集：")
    print(f"  用户数：{valid_stats['count']:,}")
    print(f"  交互总数：{valid_stats['total_interactions']:,}")
    print(f"  平均交互/用户：{valid_stats['avg_interactions']:.1f}")
    print(f"  交互范围：{valid_stats['min_interactions']} - {valid_stats['max_interactions']}")
    
    print(f"\n测试集：")
    print(f"  用户数：{test_stats['count']:,}")
    print(f"  交互总数：{test_stats['total_interactions']:,}")
    print(f"  平均交互/用户：{test_stats['avg_interactions']:.1f}")
    print(f"  交互范围：{test_stats['min_interactions']} - {test_stats['max_interactions']}")
    
    print(f"\n总计：")
    total_users = train_stats['count'] + valid_stats['count'] + test_stats['count']
    total_interactions = train_stats['total_interactions'] + valid_stats['total_interactions'] + test_stats['total_interactions']
    print(f"  用户数：{total_users:,}")
    print(f"  交互总数：{total_interactions:,}")
    print(f"  平均交互/用户：{total_interactions/total_users:.1f}")
    print(f"  物品数：{len(item_map):,}")
    
    # 估算训练时间
    print(f"\n⏱️  预估训练时间（基于默认参数）：")
    avg_iterations = train_stats['total_interactions'] / 128  # batch_size=128
    print(f"  一个 epoch 约需：{avg_iterations:,.0f} 个 iteration")
    if avg_iterations > 0:
        print(f"  每 1000 iteration 评估一次，约 {avg_iterations/1000:.1f} 次评估/epoch")
    print("="*60 + "\n")

def main():
    args = parse_args()
    
    print("="*60)
    print("🚀 数据预处理工具（增强版）")
    print("="*60)
    print(f"数据集：{args.dataset}")
    print(f"最小交互数：{args.filter_size}")
    print(f"数据分割：训练={args.train_ratio:.1%}, "
          f"验证={args.valid_ratio:.1%}, "
          f"测试={args.test_ratio:.1%}")
    if args.max_users:
        print(f"最大用户数：{args.max_users}（采样）")
    print(f"随机种子：{args.seed}")
    print("="*60 + "\n")
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 读取数据
    if args.dataset == 'book':
        source_file = 'Books_5_2023.jsonl'
        if not os.path.exists(source_file):
            print(f"❌ 错误：找不到文件 {source_file}")
            sys.exit(1)
        users, item_count = read_from_amazon(source_file)
    elif args.dataset == 'taobao':
        source_file = 'UserBehavior.csv'
        if not os.path.exists(source_file):
            print(f"❌ 错误：找不到文件 {source_file}")
            sys.exit(1)
        users, item_count = read_from_taobao(source_file)
    
    # 过滤物品
    item_map = filter_items(item_count.items(), item_count, args.filter_size)
    
    # 过滤用户
    user_ids = filter_users(users, item_map, args.filter_size)
    
    # 采样用户（如果需要）
    user_ids = sample_users(user_ids, args.max_users, args.seed)
    
    # 分割数据
    train_users, valid_users, test_users = split_users(
        user_ids, args.train_ratio, args.valid_ratio, args.test_ratio, args.seed
    )
    
    # 创建用户映射
    num_users = len(user_ids)
    user_map = dict(zip(user_ids, list(range(num_users))))
    
    # 输出统计信息
    if args.stats:
        print_statistics(users, item_map, user_map, train_users, valid_users, test_users)
    
    # 保存数据
    path = './data/' + args.dataset + '_data/'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 创建目录：{path}")
    
    print(f"\n💾 保存数据...")
    export_map(path + args.dataset + '_user_map.txt', user_map)
    export_map(path + args.dataset + '_item_map.txt', item_map)
    
    total_train = export_data(path + args.dataset + '_train.txt', train_users, 
                             users, item_map, user_map)
    total_valid = export_data(path + args.dataset + '_valid.txt', valid_users,
                             users, item_map, user_map)
    total_test = export_data(path + args.dataset + '_test.txt', test_users,
                            users, item_map, user_map)
    
    print(f"✅ 保存完成！")
    print(f"   训练集：{total_train:,} 条交互")
    print(f"   验证集：{total_valid:,} 条交互")
    print(f"   测试集：{total_test:,} 条交互")
    print(f"   总计：{total_train + total_valid + total_test:,} 条交互")
    print(f"\n📂 数据保存位置：{path}")

if __name__ == '__main__':
    main()

