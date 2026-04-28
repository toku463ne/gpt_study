import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads) == 0
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x, return_attn=False):
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, float("-inf"))

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        if return_attn:
            return context_vec, attn_weights
        return context_vec
    


import matplotlib.pyplot as plt

def plot_attention(attn_weights, tokens=None):
    """
    attn_weights: (batch, heads, tokens, tokens)
    """
    attn = attn_weights[0]  # 1バッチだけ見る
    num_heads = attn.shape[0]

    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))

    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn[h].detach().numpy())

        ax.set_title(f"Head {h}")
        ax.set_xlabel("Key (見る対象)")
        ax.set_ylabel("Query (注目する側)")

        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45)
            ax.set_yticklabels(tokens)

    plt.colorbar(im, ax=axes)
    plt.tight_layout()
    plt.show()


torch.manual_seed(42)

inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55],
])

tokens = ["Your", "journey", "starts", "with", "one", "step"]

batch = inputs.unsqueeze(0)

mha = MultiHeadAttention(
    d_in=3,
    d_out=8,          # ←ここ重要（大きくする）
    context_length=6,
    dropout=0.0,
    num_heads=2
)

out, attn = mha(batch, return_attn=True)

plot_attention(attn, tokens)