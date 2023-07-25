import os
import re
import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
# task_datasets from hf.
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *


class StreamDataset(IterableDataset):
    def __init__(self, data, tokenizer, seq_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def get_sequence(self):
        buffer_tokens = [self.tokenizer.bos_token_id]
        for x in self.data:
            curr_tokens = self.tokenizer(x['text'])['input_ids']
            buffer_tokens += curr_tokens
            while len(buffer_tokens) >= self.seq_length:
                tokens = buffer_tokens[:self.seq_length]
                buffer_tokens = [self.tokenizer.bos_token_id] + buffer_tokens[self.seq_length:]
                input_ids = torch.tensor(tokens)
                yield {
                    'text': input_ids,
                    'input_ids': input_ids,
                    'attention_mask': torch.ones_like(input_ids),
                }

    def get_stream(self):
        return cycle(self.get_sequence())

    def __iter__(self):
        return self.get_stream()


def get_bookcorpus_train_data_loader(args, tokenizer, num_workers=0):
    dataset = load_dataset('bookcorpus', split='train')
    stream_dataset = StreamDataset(dataset, tokenizer, args.seq_length)

    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader


def get_openwebtext_train_data_loader(args, tokenizer, num_workers=0):
    dataset = load_dataset('openwebtext', split='train')
    stream_dataset = StreamDataset(dataset, tokenizer, args.seq_length)

    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader


def get_wikipedia_train_data_loader(args, tokenizer, num_workers=0):
    dataset = load_dataset('wikipedia', split='train')
    stream_dataset = StreamDataset(dataset, tokenizer, args.seq_length)

    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader

