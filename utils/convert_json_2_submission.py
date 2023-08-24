import json
import csv

# 读取json文件
with open('../output/submission.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 将json数据写入csv文件
with open('../output/submission.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    
    # 写入标题行
    writer.writerow(['id', 'prediction'])
    
    # 遍历json数据，写入csv
    for item in data:
        id_ = item['id']
        prediction = ' '.join(item['output'])  # 转换列表为字符串
        writer.writerow([id_, prediction])
