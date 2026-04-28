
import torch
import torch.nn as nn

import __init__

from ch04.s4_2_layernorm import LayerNorm
from ch04.s4_5_transformer import TransformerBlock

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    batch = torch.tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 2571]])
    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput batch:\n", out.shape)
    print(out)

    """
    Input batch:
 tensor([[6109, 3626, 6100,  345],
        [6109, 1110, 6622, 2571]])

Output batch:
 torch.Size([2, 4, 50257])
tensor([[[ 0.1380,  0.0079, -0.1958,  ..., -0.0223, -0.1062,  0.1717],
         [ 0.3866, -0.8400, -0.6559,  ..., -0.5162,  0.2361, -0.3350],
         [ 0.6984, -0.1825, -0.1633,  ...,  0.1471, -0.6503, -0.0054],
         [-0.4288,  0.1671, -0.1261,  ...,  1.1572,  0.5297, -0.5542]],

        [[ 0.1094, -0.2890, -0.1463,  ..., -0.0558,  0.2907, -0.2818],
         [ 0.0883, -0.3544, -0.3523,  ...,  1.2921,  0.0050,  0.1902],
         [ 0.6092,  0.4702, -0.4092,  ...,  0.7682,  0.3781, -0.1969],
         [-0.1772, -0.6757, -0.5362,  ...,  1.1663, -0.2468, -0.6269]]],
       grad_fn=<UnsafeViewBackward0>)
    """

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    """
    Total number of parameters: 163,009,536
    """