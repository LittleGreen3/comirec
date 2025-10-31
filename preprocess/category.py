import sys
import json

name = 'book'
if len(sys.argv) > 1:
    name = sys.argv[1]

item_cate = {}
item_map = {}
cate_map = {}
with open('./data/%s_data/%s_item_map.txt' % (name, name), 'r', encoding='utf-8') as f:
    for line in f:
        conts = line.strip().split(',')
        item_map[conts[0]] = conts[1]

if name == 'taobao':
    with open('UserBehavior.csv', 'r', encoding='utf-8') as f:
        for line in f:
            conts = line.strip().split(',')
            iid = conts[1]
            if conts[3] != 'pv':
                continue
            cid = conts[2]
            if iid in item_map:
                if cid not in cate_map:
                    cate_map[cid] = len(cate_map) + 1
                item_cate[item_map[iid]] = cate_map[cid]
elif name == 'book':
    with open('meta_Books.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                iid = r['parent_asin']
                
                # 检查是否有 category 字段
                if 'categories' not in r:
                    continue
                
                cates = r['categories']
                
                # 检查 category 是否为空列表
                if not cates:
                    continue
                
                if iid not in item_map:
                    continue
                
                # 使用最后一个元素作为最具体的类别
                cate = cates[-1]
                if cate not in cate_map:
                    cate_map[cate] = len(cate_map) + 1
                item_cate[item_map[iid]] = cate_map[cate]
            except (KeyError, IndexError, ValueError, TypeError) as e:
                # 跳过格式不正确或缺少字段的行
                continue

with open('./data/%s_data/%s_cate_map.txt' % (name, name), 'w', encoding='utf-8') as f:
    for key, value in cate_map.items():
        f.write('%s,%s\n' % (key, value))
with open('./data/%s_data/%s_item_cate.txt' % (name, name), 'w', encoding='utf-8') as f:
    for key, value in item_cate.items():
        f.write('%s,%s\n' % (key, value))
