# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch
import random
from torch.utils.data import Dataset
from typing import List

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

prompt_template = (
    B_SYS + "Below is an instruction that describes a task. " +
    "Write a response that appropriately completes the request." + E_SYS +
    "### Instruction:\n{}\n\n### Response:\n"
)

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_tokens=2048):
        # 存储数据
        positive_parsed_data = []
        negative_parsed_data = []
        train_parsed_data = []
        test_parsed_data = []

        # 读取 merged_dataset.jsonl
        file_path = "/home/jli265/workspace/llama-recipes/misc/deepinception_special_dataset.jsonl"

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                data_entry = {
                    "conversations": [obj.get("query", ""), obj.get("response", "")],
                    "source": obj.get("from", ""),
                    "label": obj.get("label", 0)  # 取 label 字段
                }
                if obj.get("label", 0) == 1:  # 1 代表正样本, malicious
                    positive_parsed_data.append(data_entry)
                else:  # 0 代表负样本, benign
                    negative_parsed_data.append(data_entry)

        random.shuffle(positive_parsed_data)
        random.shuffle(negative_parsed_data)

        # 计算 0.5% 的抽样数量
        # pos_sample_size = max(1, int(len(positive_parsed_data) * 0.005))  # 至少抽 1 个
        # neg_sample_size = max(1, int(len(negative_parsed_data) * 0.005))  # 至少抽 1 个

        # 进行抽样
        # test_positive_samples = random.sample(positive_parsed_data, pos_sample_size)
        # test_negative_samples = random.sample(negative_parsed_data, neg_sample_size)

        # 将抽取的样本加入测试集
        # test_parsed_data.extend(test_positive_samples)
        # test_parsed_data.extend(test_negative_samples)

        # 从原数据集中移除测试样本，确保训练数据与测试数据不重叠
        # positive_parsed_data = [d for d in positive_parsed_data if d not in test_positive_samples]
        # negative_parsed_data = [d for d in negative_parsed_data if d not in test_negative_samples]

        train_parsed_data.extend(positive_parsed_data)
        train_parsed_data.extend(negative_parsed_data)

        # 打印数据统计信息
        print(f"📊 Positive samples: {len(positive_parsed_data)}")
        print(f"📊 Negative samples: {len(negative_parsed_data)}")
        print(f"📊 Test samples: {len(test_parsed_data)}")

        random.shuffle(train_parsed_data)
        # random.shuffle(test_parsed_data)

        if partition == "train_no_safety.jsonl":
            self.ann = train_parsed_data
        elif partition == "test_no_safety.jsonl":
            self.ann = test_parsed_data
        else:
            assert False, "temporaily not support"

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        conversation_0 = ann["conversations"][0]  # Extract only `conversations[0]`
        conversation_1 = ann["conversations"][1]  # Extract only `conversations[1]`
        prompt = B_INST + " " + prompt_template.format(conversation_0) + " " + E_INST
        example = prompt + " " + conversation_1 + " "

        # Tokenize `conversations[0]` separately to get specific indices
        conversation_0_tokens = self.tokenizer.encode(conversation_0)
        conversation_1_tokens = self.tokenizer.encode(conversation_1)

        # Tokenize the prompt and example
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_tokens.insert(0, self.tokenizer.cls_token_id)  # Add CLS token at the start
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.int64)

        example_tokens = self.tokenizer.encode(example)
        example_tokens.append(self.tokenizer.eos_token_id)  # Add EOS token at the end
        example_tokens.insert(0, self.tokenizer.cls_token_id)  # Add CLS token at the start
        example_tensor = torch.tensor(example_tokens, dtype=torch.int64)

        # Calculate padding
        padding = self.max_tokens - example_tensor.shape[0]
        if padding < 0:
            example_tensor = example_tensor[:self.max_tokens]

        # Create labels and mask out prompt tokens in labels
        labels = copy.deepcopy(example_tensor)
        labels[: len(prompt_tensor)] = -1

        # Find `conversations[0]` start and end indices in `input_ids`
        start_idx_input = len(self.tokenizer.encode(B_INST + " " + prompt_template.format(""))) + 1
        end_idx_input = start_idx_input + len(conversation_0_tokens) - 1

        # Find `conversations[1]` start and end indices in `input_ids`
        start_idx_output = len(prompt_tokens)
        if ann["label"] == 1:
            end_idx_output = start_idx_output + len(conversation_1_tokens) - 1
        else:  # If label is 0, make end_idx_output equal to start_idx_output
            end_idx_output = start_idx_output

        # Generate attention and label masks
        example_mask = example_tensor.ge(0)
        label_mask = labels.ge(0)
        example_tensor[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        labels[0] = ann["label"]
        # Pack positions into a list
        positions = [start_idx_input, end_idx_input, start_idx_output, end_idx_output]

        return {
            "input_ids": example_tensor.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
            "positions": positions  # Packed position indices
        }
