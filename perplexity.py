from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import math

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IndexDataset(Dataset):
    """将长文本按固定长度分块并存储为 PyTorch Dataset"""
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    """
    将文本样本拼接成一个长字符串，分词后按照 seq_len 划分成块。
    
    Args:
        samples: 数据样本列表
        tokenizer: 分词器
        seq_len: 块的最大长度
        field_name: 样本的字段名
    
    Returns:
        IndexDataset: 数据块组成的 PyTorch Dataset
    """
    # 拼接长文本
    full_text = "\n\n".join(samples[field_name] if field_name else samples)
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    
    # 划分成固定长度的块
    num_chunks = input_ids.numel() // seq_len
    chunks = [input_ids[i * seq_len:(i + 1) * seq_len] for i in range(num_chunks)]
    return IndexDataset(torch.stack(chunks))

def get_wikitext2(tokenizer, seq_len, split="test"):
    """
    加载 WikiText-2 数据集，并按照 seq_len 处理。
    
    Args:
        tokenizer: 分词器
        seq_len: 块的最大长度
        split: 数据集划分 ("train", "test")
    
    Returns:
        Dataset: 处理后的数据块
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return process_data(dataset, tokenizer, seq_len, "text")

def get_ptb(tokenizer, seq_len, split="test"):
    """
    加载 PTB 数据集，并按照 seq_len 处理。
    
    Args:
        tokenizer: 分词器
        seq_len: 块的最大长度
        split: 数据集划分 ("train", "validation")
    
    Returns:
        Dataset: 处理后的数据块
    """
    dataset = load_dataset("ptb_text_only", "penn_treebank", split=split)
    return process_data(dataset, tokenizer, seq_len, "sentence")

def calculate_perplexity(model, data_loader):
    """
    计算模型在给定数据集上的困惑度。
    
    Args:
        model: 预训练模型
        data_loader: 数据加载器 (DataLoader)
    
    Returns:
        perplexity: 困惑度值
    """
    model.eval()
    model.to(device)
    
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = batch.to(device)
            labels = batch.clone()
            outputs = model(batch, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * batch.size(1)
            total_tokens += batch.size(1)
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

def evaluate_perplexity(model_name, dataset_name, seq_len=1024, batch_size=8):
    """
    加载模型和数据集，评估困惑度。
    
    Args:
        model_name: 模型名称
        dataset_name: 数据集名称 ("wikitext2" 或 "ptb")
        seq_len: 最大序列长度
        batch_size: 批大小
    
    Returns:
        perplexity: 困惑度值
    """
    print(f"加载模型 {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    print(f"加载数据集 {dataset_name}...")
    if dataset_name == "wikitext2":
        dataset = get_wikitext2(tokenizer, seq_len, split="test")
    elif dataset_name == "ptb":
        dataset = get_ptb(tokenizer, seq_len, split="validation")
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    # 构建 DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"开始评估模型 {model_name} 在数据集 {dataset_name} 上的困惑度...")
    perplexity = calculate_perplexity(model, data_loader)
    print(f"模型 {model_name} 在数据集 {dataset_name} 上的困惑度为: {perplexity:.2f}")
    return perplexity

if __name__ == "__main__":
    # 配置
    models = ["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-Instruct-v0.2"]
    datasets = ["wikitext2", "ptb"]
    seq_len = 2048
    batch_size = 8

    results = {}
    for model_name in models:
        results[model_name] = {}
        for dataset_name in datasets:
            perplexity = evaluate_perplexity(model_name, dataset_name, seq_len, batch_size)
            results[model_name][dataset_name] = perplexity

    print("\n最终困惑度结果:")
    for model_name, dataset_results in results.items():
        print(f"模型 {model_name}:")
        for dataset_name, perplexity in dataset_results.items():
            print(f"  {dataset_name}: {perplexity:.2f}")
