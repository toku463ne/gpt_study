import torch
import tiktoken

import __init__

from ch04.s4_6_gptmodel import GPTModel, GPT_CONFIG_124M


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

if __name__ == "__main__":
    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    print("Encoded input:", encoded)
    # [15496, 11, 314, 716]

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("Input tensor shape:", encoded_tensor.shape)
    # [1, 4]

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    out = generate_text_simple(model, encoded_tensor, 
                               max_new_tokens=6, 
                               context_size=GPT_CONFIG_124M["context_length"])
    print("Output:", out)
    # Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])

    print("Output length:", len(out[0]))
    # Output length: 10

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("Decoded output:", decoded_text)
    # Decoded output: Hello, I am Featureiman Byeswickattribute argue
