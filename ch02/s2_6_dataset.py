import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


import __init__


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, 
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def sample1():
    with open("ch02/edith-wharton.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=1,
        max_length=4,
        stride=1,
        shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print("Input ID:", first_batch)
"""
Input ID: [tensor([[  39, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
"""


def sample2():
    with open("ch02/edith-wharton.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Input IDs:\n", inputs)
    print("Target IDs:\n", targets)

"""
Input IDs:
 tensor([[   39,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])
Target IDs:
 tensor([[ 2885,  1464,  1807,  3619],
        [  402,   271, 10899,  2138],
        [  257,  7026, 15632,   438],
        [ 2016,   257,   922,  5891],
        [ 1576,   438,   568,   340],
        [  373,   645,  1049,  5975],
        [  284,   502,   284,  3285],
        [  326,    11,   287,   262]])
"""
