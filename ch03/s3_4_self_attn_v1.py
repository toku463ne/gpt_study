import torch.nn as nn
import torch

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Parameter(torch.rand(d_in, d_out))
        self.W_k = nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        k = x @ self.W_k
        q = x @ self.W_q
        v = x @ self.W_v

        attn_scores = q @ k.T 
        attn_weights = torch.softmax(attn_scores / k.shape[1]**0.5, dim=-1)
        context_vector = attn_weights @ v
        return context_vector
    

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts (x^3)
        [0.22, 0.58, 0.33],  # with (x^4)
        [0.77, 0.25, 0.10],  # one (x^5)
        [0.05, 0.80, 0.55],  # step (x^6)
    ]
)

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in=inputs.shape[1], d_out=2)
print(sa_v1(inputs))