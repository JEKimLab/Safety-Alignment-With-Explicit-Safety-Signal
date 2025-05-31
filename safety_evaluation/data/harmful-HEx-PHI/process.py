import json
import csv
from transformers import LlamaTokenizer

# 初始化 Llama2-7B 的 tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 定义输入和输出文件路径
input_file = "./Harmful-HEx-PHI.jsonl"
output_files = {
    5: "hex_prefill_qi_5.csv",
    10: "hex_prefill_qi_10.csv",
    20: "hex_prefill_qi_20.csv",
    40: "hex_prefill_qi_40.csv",
}

# 读取 JSONL 文件并处理
def process_jsonl(input_file, output_files):
    # 创建 CSV 文件写入器
    writers = {}
    for num_tokens, output_file in output_files.items():
        writers[num_tokens] = csv.writer(open(output_file, "w", newline='', encoding="utf-8"), delimiter=',', quoting=csv.QUOTE_ALL)

    # 逐行读取 JSONL 文件
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            # 每行是一个数组，解析 JSON 数组
            data = json.loads(line.strip())
            
            # 提取 user 和 assistant 的内容
            user_content = None
            assistant_content = None
            
            for entry in data:
                if entry.get("role") == "user":
                    user_content = entry.get("content", "")
                elif entry.get("role") == "assistant":
                    assistant_content = entry.get("content", "")

            if user_content and assistant_content:
                # 对 assistant 内容进行 tokenization
                token_ids = tokenizer.encode(assistant_content, add_special_tokens=False)

                # 提取前 N 个 token，并解码为字符串
                for num_tokens, writer in writers.items():
                    truncated_token_ids = token_ids[:num_tokens]
                    truncated_content = tokenizer.decode(truncated_token_ids)

                    # 写入 CSV 文件
                    writer.writerow([user_content, truncated_content])

# 执行处理
process_jsonl(input_file, output_files)
