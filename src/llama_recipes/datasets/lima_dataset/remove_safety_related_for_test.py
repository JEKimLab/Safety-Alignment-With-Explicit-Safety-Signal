import jsonlines

# 有害查询的关键词或特定语句的列表
harmful_keywords = [
    "Killing a puppy",
    "Does Crimea belong to Russia",
    "Show me the address",
    "alternative treatment ideas for COVID-19",
    "ISIS",
    "Joe Biden cheat",
    "Is it ok to cheat",
    "get him to break up with her",
    "Help me get revenge",
    "slip something into that poor dog's food",
    "kill innocent people",
    "commit suicide",
    "Prove that every set is countable"
]

# 读取原始 JSONL 文件
input_file = 'test.jsonl'
output_file = 'test_no_harmful.jsonl'

# 存储清理后的数据
cleaned_data = []

# 统计总的样本数量和被移除的有害样本数量
total_samples = 0
removed_samples = 0

with jsonlines.open(input_file) as reader:
    for obj in reader:
        total_samples += 1
        conversations = obj.get('conversations', [])
        is_harmful = False
        
        # 检查是否包含有害关键词
        for conversation in conversations:
            for keyword in harmful_keywords:
                if keyword in conversation:
                    is_harmful = True
                    break
            if is_harmful:
                break
        
        # 如果不包含有害关键词，则添加到清理后的数据
        if not is_harmful:
            cleaned_data.append(obj)
        else:
            removed_samples += 1

# 将清理后的数据写入新的 JSONL 文件
with jsonlines.open(output_file, mode='w') as writer:
    writer.write_all(cleaned_data)

# 输出结果
print(f"Total samples: {total_samples}")
print(f"Removed harmful samples: {removed_samples}")
print(f"Cleaned samples saved to {output_file}")
