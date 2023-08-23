import json
import random

with open('data/LLMScience.json',encoding='utf-8') as f:
    datas = json.load(f)

datas = datas[:200]

# datas = random.sample(datas, 5)

prompt = """Question: {}
A: {}
B: {}
C: {}
D: {}
E: {}
Answer: {}"""

# 筛选出所有答案为A,B,C,D,E的数据
data_A = [item for item in datas if item['answer'] == 'A']
data_B = [item for item in datas if item['answer'] == 'B']
data_C = [item for item in datas if item['answer'] == 'C']
data_D = [item for item in datas if item['answer'] == 'D']
data_E = [item for item in datas if item['answer'] == 'E']

# 检查每个答案是否都至少有一个数据
if not all([data_A, data_B, data_C, data_D, data_E]):
    print("数据中不包含所有的答案至少一次!")
else:
    sampled_data = []
    sampled_data.append(random.choice(data_A))
    sampled_data.append(random.choice(data_B))
    sampled_data.append(random.choice(data_C))
    sampled_data.append(random.choice(data_D))
    sampled_data.append(random.choice(data_E))

random.shuffle(sampled_data)

total_few_shot = ""
for data in sampled_data:
    question = data['prompt']
    A = data['A']
    B = data['B']
    C = data['C']
    D = data['D']
    E = data['E']
    answer = data['answer']
    total_few_shot+=prompt.format(question,A,B,C,D,E,answer)+"\n\n"

print(total_few_shot)


