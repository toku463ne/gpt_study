import torch

import __init__
from ch02.s2_2_read_verdict import *
from ch02.s2_6_dataset import *

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4

dataloader = create_dataloader_v1(
    text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)

data_iter = iter(dataloader) ## DataLoader.__iter__() is defined in pytorch
inputs, targets = next(data_iter) 
"""
calls data_iter.__next__()

1. get next batch indices
2. call dataset.__getitem__ for those indices
3. collate them into tensors
4. return batch
"""

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
"""
torch.Size([8, 4])
"""

token_embeddings = token_embedding_layer(inputs)
# same as token_embeddings = token_embedding_layer.weight[inputs]
print(token_embeddings.shape)
"""
torch.Size([8, 4, 256])
"""

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# get embeddings for positions 0, 1, 2, ..., context_length-1
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
"""
torch.Size([4, 256])
"""