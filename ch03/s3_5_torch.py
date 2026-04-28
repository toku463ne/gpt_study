import torch

t = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=torch.float32)


a = torch.triu(torch.ones(3, 3), diagonal=1)
#print(a)


t.masked_fill_(a.bool(), float('-inf'))
print(t)
"""
tensor([[1., -inf, -inf],
        [1., 2., -inf],
        [1., 2., 3.]])
"""


print(a)

