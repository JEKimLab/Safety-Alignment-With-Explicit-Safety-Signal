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
# SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
# SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS

prompt_template = (
    B_SYS + "Below is an instruction that describes a task. " +
    "Write a response that appropriately completes the request." + E_SYS +
    "### Instruction:\n{}\n\n### Response:\n"
)

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_tokens=2048):
        parsed_data = []
        test_parsed_data = []
        
        with open(f"{dataset_config.data_path}/{dataset_config.train_split}", 'r', encoding='utf-8') as f:
            for line in f:
                # Parse each line as a JSON object
                obj = json.loads(line.strip())
                conversations = obj.get("conversations", [])
                source = obj.get("source", "")
                
                # Append the parsed conversation and source to the list
                parsed_data.append({
                    "conversations": conversations,
                    "source": source,
                    "label": 1 # safe query
                })

        alpaca_no_safety_datasets = json.load(open("src/llama_recipes/datasets/alpaca_data_no_safety.json"))
        for sample in alpaca_no_safety_datasets:
            if not sample["input"]: # to compatible with alert dataset, we exclude samples with input
                if len(parsed_data) < 14763: # number of alert sft dataset
                    parsed_data.append({
                        "conversations": [sample["instruction"], sample["output"]],
                        "source": "alpaca_data_no_safety",
                        "label": 1 # safe query
                    }) 
                else:
                    test_parsed_data.append({
                        "conversations": [sample["instruction"], sample["output"]],
                        "source": "alpaca_data_no_safety",
                        "label": 1 # safe query
                    })

                if len(test_parsed_data) > int(len(alpaca_no_safety_datasets) / 20):
                    break

        num_positive_smaples = len(parsed_data)
        print("Number of positive smaples:", num_positive_smaples)
        with open("src/llama_recipes/datasets/alert_sft/alert_sft_chosen.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                parsed_data.append({
                    "conversations": [obj.get("prompt", ""),obj.get("response", "")],
                    "source": obj.get("category", ""),
                    "label": 0 # query with malicious intention
                })

        print("Number of negative smaples:", len(parsed_data) - num_positive_smaples)
        print("Number of test smaples:", len(test_parsed_data))

        random.shuffle(parsed_data)
        if partition == "train_no_safety.jsonl":
            self.ann = parsed_data
        elif partition == "test_no_safety.jsonl":
            self.ann = test_parsed_data
        else:
            assert False, "temporaily not support"

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.ann)

    # def __getitem__(self, index):
    #     IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

    #     ann = self.ann[index]
    #     prompt = B_INST + " " + prompt_template.format(ann["conversations"][0]) + " " + E_INST
    #     example = prompt + " " + ann["conversations"][1] + " "
    #     prompt = self.tokenizer.encode(prompt)
    #     prompt.insert(0, self.tokenizer.cls_token_id) 
    #     prompt = torch.tensor(
    #         prompt, dtype=torch.int64
    #     )

    #     example = self.tokenizer.encode(example)
    #     example.append(self.tokenizer.eos_token_id)
    #     example.insert(0, self.tokenizer.cls_token_id)
    #     example = torch.tensor(
    #         example, dtype=torch.int64
    #     )
        
    #     padding = self.max_tokens - example.shape[0]
    #     if padding < 0:
    #         example = example[: self.max_tokens]

    #     labels = copy.deepcopy(example)
    #     labels[: len(prompt)] = -1
    #     example_mask = example.ge(0)
    #     label_mask = labels.ge(0)
    #     example[~example_mask] = 0
    #     labels[~label_mask] = IGNORE_INDEX
    #     labels[0] = ann["label"]

    #     return {
    #         "input_ids": example.tolist(),
    #         "labels": labels.tolist(),
    #         "attention_mask":example_mask.tolist(),
    #     }

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

