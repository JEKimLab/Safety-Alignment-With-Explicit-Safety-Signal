import json
import csv

# 输入输出文件路径
input_file = '/home/jli265/workspace/llama-recipes/safety_evaluation/data/adv_bench_sub_llama2.jsonl'
output_file = 'adv_bench_sub_llama2.csv'

# 打开输出文件，设置正确的 CSV 写入参数
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

    # 逐行读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                prompt = obj.get('ss_prompt', '')
                writer.writerow([prompt])
            except json.JSONDecodeError:
                continue  # 忽略格式错误的行
