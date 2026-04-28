import torch.nn as nn
import torch

k = nn.Linear(3, 2)
print(k.weight)
"""
tensor([[-0.1908,  0.4300, -0.5704],
        [ 0.4238,  0.0781,  0.0666]], requires_grad=True)
"""

print(k.bias)
"""
tensor([-0.0865,  0.0312], requires_grad=True)
"""


x = torch.tensor([0.43, 0.15, 0.89])
print(k(x))
## same as y = x @ W.T + b
"""
tensor([-0.6116,  0.2844], grad_fn=<ViewBackward0>)
"""