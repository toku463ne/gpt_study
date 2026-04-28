import torch

import __init__


vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
"""
# Similar to below
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )

    def forward(self, input):
        return F.embedding(input, self.weight)
"""


#print(embedding_layer.weight)
"""
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
"""

#print(embedding_layer(torch.tensor([[3]])))
"""
tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
"""


input_ids = torch.tensor([2,3,5,1])
#print(embedding_layer(input_ids))
"""
tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
"""

output = embedding_layer.weight[input_ids]
print(output)
"""
same as above
"""




x = embedding_layer(torch.tensor([3]))
y = embedding_layer(torch.tensor([[3]]))
z = embedding_layer(torch.tensor([[[3]]]))

#print(embedding_layer.weight.shape)
#print(x.shape)
#print(y.shape)
#print(z.shape)
"""
torch.Size([6, 3])
torch.Size([1, 3])
torch.Size([1, 1, 3])
"""