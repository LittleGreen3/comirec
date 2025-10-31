import os

def view_taobao_behavior_structure():
    # 定义文件路径：从preprocess文件夹向上一级（项目根目录），再进入data/front_taobao
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # 当前脚本所在目录（preprocess）
        "..",  # 上一级目录（项目根目录）
        "data", 
        "front_taobao", 
        "UserBehavior.csv"
    )
    # 标准化路径（处理../等相对路径）
    file_path = os.path.normpath(file_path)
    
    try:
        # 读取前10行数据
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"文件路径：{file_path}\n")
            print("前10行数据（每行格式：用户ID,物品ID,类别ID,行为类型,时间戳）：")
            print("-" * 80)
            for i in range(10):
                line = f.readline()
                if not line:  # 文件不足10行时退出
                    break
                # 去除换行符并打印
                print(f"第{i+1}行：{line.strip()}")
            print("-" * 80)
            print("\n结构说明：")
            print("淘宝UserBehavior.csv通常包含5个字段（逗号分隔）：")
            print("1. 用户ID（整数）")
            print("2. 物品ID（整数）")
            print("3. 类别ID（整数）")
            print("4. 行为类型（字符串，如'pv'浏览、'buy'购买等）")
            print("5. 时间戳（整数，秒级时间）")
    
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        print("请检查文件路径是否正确，确保data/front_taobao/UserBehavior.csv存在")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")

if __name__ == "__main__":
    view_taobao_behavior_structure()