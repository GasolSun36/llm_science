import json

with open('../output/submission.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('../data/official_llm_train.json', 'r', encoding='utf-8') as file:
    answers = json.load(file)

def compute_map_3(output, answer):
    # 初始化一个总的Precision值
    total_precision = 0

    # 检查前三个位置中是否有正确答案
    for k in range(1, 4):
        if k <= len(output) and output[k-1] == answer:
            total_precision += 1 / k

    # 计算MAP@3
    map_3 = total_precision
    return map_3

accuracy = 0
for i in range(len(data)):
    output = data[i]['output']
    answer = answers[i]['answer']
    accuracy += compute_map_3(output, answer)

print(f"Top-3 Accuracy: {float(accuracy/len(answers)) * 100}%")  # testset acc: 69.92