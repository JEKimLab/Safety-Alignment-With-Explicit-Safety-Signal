############ GPT-4 Judge ##################


import os
import fire
import csv
import json
import numpy as np
from eval_utils.openai_gpt4_judge import duo_judge
import time

import json

def extract_qam(log_file_path, output_file_path=None):
    """
    解析日志文件，提取 Q, A, M 并返回列表，同时可选保存到 JSON 文件。
    
    :param log_file_path: 日志文件路径
    :param output_file_path: 保存 JSON 文件路径 (可选)
    :return: 提取的 QAM 列表
    """
    # 初始化一个空列表用于存储结果
    results = []

    # 读取日志文件
    with open(log_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 初始化临时变量用于存储 Q, A 和 M
    current_q = None
    current_a = None
    is_reading_q = False
    is_reading_a = False

    # 遍历日志文件的每一行
    for line in lines:
        stripped_line = line.strip()

        # 检测 Prompt (Q)
        if "Prompt:" in stripped_line:
            # 如果之前的 QAM 已经完整，则保存到结果中
            if current_q and current_a:
                results.append({
                    "Q": current_q.strip(), 
                    "A": current_a.strip().split("[/INST]")[1].strip(), 
                    "M": "unsafe"})
                current_q, current_a, current_m = None, None, None
                is_reading_q = False
                is_reading_a = False
            
            # 提取新的 Q
            prompt_start = stripped_line.find("Prompt:") + len("Prompt:")
            current_q = stripped_line[prompt_start:].strip()
            is_reading_q = True
            is_reading_a = False

        # 检测 Answer (A)
        elif "Answer:" in stripped_line:
            is_reading_a = True
            is_reading_q = False
            answer_start = stripped_line.find("Answer:") + len("Answer:")
            current_a = stripped_line[answer_start:].strip()

        # 检测 Moderation Result (M)
        elif "Moderation Result:" in stripped_line:
            is_reading_a = False  # 停止读取 Answer

        # 继续读取多行 Q
        elif is_reading_q:
            current_q += " " + stripped_line

        # 继续读取多行 A
        elif is_reading_a:
            current_a += " " + stripped_line
        
    # 如果最后还有未保存的 QAM，则添加到结果中
    if current_q and current_a:
        results.append({"Q": current_q.strip(), "A": current_a.strip(), "M": "unsafe"})

    # 如果指定了输出文件路径，则将结果保存为 JSON
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file, indent=4, ensure_ascii=False)
        print(f"Extracted {len(results)} QAM blocks. Results saved to {output_file_path}")

    # 返回提取的 QAM 列表
    return results



def main(
    input_file="",
    output_file="",
    **kwargs
):
    # 示例用法
    QAMpairs = extract_qam(input_file)

    unsafes, responses, effective_QAparits = duo_judge(
        [(qam["Q"], qam["A"], qam["M"]) for qam in QAMpairs]) # Call GPT4 API
    
    # Save results
    qa_records = []
    cnt = 0
    for (que, ans, _) in effective_QAparits:
        record = {
            'que': que,
            'ans': ans,
            'unsafe': unsafes[cnt],
            'duo_reason': responses[cnt],
        }
        qa_records.append(record)
        cnt += 1
        
    real_unsafe_number = len([tag for tag in unsafes if "Yes" in tag])
    print(f"There is {real_unsafe_number} real unsafe response in total {len(unsafes)} samples")

    # Dump to output file
    output_file = f"safety_evaluation/gpt4_eval_output/{output_file}_output.jsonl"
    with open(output_file, 'w') as f:
        for li in qa_records:
            f.write(json.dumps(li))
            f.write("\n")
    print(f"Detailed results (scores and resons) are saved to {output_file}.")

if __name__ == "__main__":
    fire.Fire(main)