import json
from collections import defaultdict

# 第一步：统计每个用户的评论数量
print("正在统计用户评论数量...")
user_review_count = defaultdict(int)

with open('Books.jsonl', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        if line_num % 100000 == 0:
            print(f"已处理 {line_num} 行...")
        try:
            data = json.loads(line.strip())
            user_id = data.get('user_id')
            if user_id:
                user_review_count[user_id] += 1
        except json.JSONDecodeError:
            continue

print(f"统计完成，共 {len(user_review_count)} 个用户")

# 筛选出评论数>=5的用户
filtered_users = {user_id for user_id, count in user_review_count.items() if count >= 5}
print(f"评论数>=5的用户数量: {len(filtered_users)}")

# 第二步：提取这些用户的所有评论数据
print("正在提取用户评论数据...")
filtered_reviews = []

with open('Books.jsonl', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        if line_num % 100000 == 0:
            print(f"已处理 {line_num} 行，已提取 {len(filtered_reviews)} 条评论...")
        try:
            data = json.loads(line.strip())
            user_id = data.get('user_id')
            if user_id and user_id in filtered_users:
                filtered_reviews.append(data)
        except json.JSONDecodeError:
            continue

print(f"提取完成，共 {len(filtered_reviews)} 条评论")

# 第三步：保存为JSONL文件
print("正在保存到 Books_5_2023.jsonl...")
with open('Books_5_2023.jsonl', 'w', encoding='utf-8') as f:
    for review in filtered_reviews:
        f.write(json.dumps(review, ensure_ascii=False) + '\n')

print(f"完成！已保存 {len(filtered_reviews)} 条评论到 Books_5_2023.jsonl")

