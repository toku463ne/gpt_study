import torch
import torch.nn as nn

# nn.Linear(入力のサイズ, 出力のサイズ)
layer = nn.Linear(in_features=128, out_features=10)
# y = xA^T + b


# ダミーデータの作成 (バッチサイズ 32, 特徴量 128)
input_data = torch.randn(32, 128)

# 実行
output = layer(input_data)

print(output.shape) # torch.Size([32, 10]) に変換される
