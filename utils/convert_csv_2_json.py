import csv
import json

# 定义CSV文件和将要创建的JSON文件的名称
csv_filename = '../data/train.csv'
json_filename = '../data/llm_train.json'

# 读取CSV并转换为JSON
data = []
with open(csv_filename, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append(row)

# 保存为JSON文件
with open(json_filename, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f'{csv_filename} has been converted to {json_filename}')
