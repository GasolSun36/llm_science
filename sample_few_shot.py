import json
import random

with open('data/LLMScience.json',encoding='utf-8') as f:
    datas = json.load(f)

datas = datas[:200]

datas = random.sample(datas, 5)

prompt = """Question: {}
A: {}
B: {}
C: {}
D: {}
E: {}
Answer: {}"""

total_few_shot = ""
for data in datas:
    question = data['prompt']
    A = data['A']
    B = data['B']
    C = data['C']
    D = data['D']
    E = data['E']
    answer = data['answer']
    total_few_shot+=prompt.format(question,A,B,C,D,E,answer)+"\n\n"

print(total_few_shot)


