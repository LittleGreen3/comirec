import json
import ijson
from decimal import Decimal

# 自定义JSON编码器，处理Decimal类型
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

# 读取JSON格式的文件
input_file = 'Books_5_2023.json'
output_file = 'Books_5_2023.jsonl'

print(f"正在流式处理 {input_file}...")

count = 0
with open(input_file, 'rb') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    # 使用ijson流式解析JSON数组中的每个对象
    parser = ijson.items(f_in, 'item')
    for item in parser:
        count += 1
        if count % 10000 == 0:
            print(f"已处理 {count} 条记录...")
        f_out.write(json.dumps(item, ensure_ascii=False, cls=DecimalEncoder) + '\n')

print(f"完成！已转换并保存 {count} 条记录到 {output_file}")

