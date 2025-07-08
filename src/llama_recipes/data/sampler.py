# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import random
from itertools import islice

import numpy as np
import torch


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)

class LengthBasedOverSamplingBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # 确定 key
        first_key = "input_ids"
        label_key = "labels"  # 你的 label 存在这里

        # 获取所有数据的长度和 label（取 label 的第一个值）
        self.lengths = [len(d[first_key]) for d in data_source]
        self.labels = [d[label_key][0] for d in data_source]

        # 按 label 分类
        self.pos_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.neg_indices = [i for i, label in enumerate(self.labels) if label == 0]

        # 计算正负比例，并对正类过采样
        pos_count, neg_count = len(self.pos_indices), len(self.neg_indices)
        print(f"pos_count: {pos_count}, neg_count: {neg_count}")
        if pos_count > 0:
            self.oversampled_pos_indices = np.random.choice(self.pos_indices, size=neg_count, replace=True).tolist()
        else:
            self.oversampled_pos_indices = []

        # 合并新的索引
        self.balanced_indices = self.oversampled_pos_indices + self.neg_indices

        # 按长度排序
        self.balanced_indices.sort(key=lambda i: self.lengths[i])

    def __iter__(self):
        ids = np.array(self.balanced_indices)

        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i + self.batch_size].tolist() for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.balanced_indices) // self.batch_size
        else:
            return len(self.balanced_indices) // self.batch_size + (len(self.balanced_indices) % self.batch_size > 0)


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
        # self.batch_sampler = LengthBasedOverSamplingBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
            )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas
