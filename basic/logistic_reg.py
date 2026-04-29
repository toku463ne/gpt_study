import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 小さめのサンプルデータの作成
# ==========================================
np.random.seed(0)
torch.manual_seed(0)

# クラス0のデータ (左下に分布する青い点)
X0 = np.random.randn(50, 2) + np.array([-1.5, -1.5])
y0 = np.zeros(50)

# クラス1のデータ (右上に分布する赤い点)
X1 = np.random.randn(50, 2) + np.array([1.5, 1.5])
y1 = np.ones(50)

# データを結合してPyTorchのTensorに変換
X = np.vstack([X0, X1]).astype(np.float32)
y = np.concatenate([y0, y1]).astype(np.float32).reshape(-1, 1)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# ==========================================
# 2. モデルの定義 (ロジスティック回帰)
# ==========================================
# 入力が2次元（Feature 1, 2）、出力が1次元（確率）
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=1),
    nn.Sigmoid()
)

# ==========================================
# 3. 学習の準備と実行
# ==========================================
# model.parameters() の中身をループで取り出す
for i, param in enumerate(model.parameters()):
    print(f"パラメータ {i}:")
    print(f"  形状: {param.shape}")
    print(f"  値: \n{param.data}")
"""
パラメータ 0:
  形状: torch.Size([1, 2])
  値: tensor([[-0.0053,  0.3793]])
パラメータ 1:
  形状: torch.Size([1])
  値: tensor([-0.5820])
"""

criterion = nn.BCELoss() # バイナリクロスエントロピー誤差
optimizer = optim.SGD(model.parameters(), lr=0.1) # 最適化手法 (SGD)

losses = []
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()      # 勾配のリセット
    outputs = model(X_tensor)  # 予測
    loss = criterion(outputs, y_tensor) # 誤差の計算
    loss.backward()            # 逆伝播（微分の計算）
    optimizer.step()           # 重みの更新
    
    losses.append(loss.item()) # グラフ用に誤差を記録

# ==========================================
# 4. 結果のプロット (グラフの描画)
# ==========================================
plt.figure(figsize=(12, 5))

# Lossの推移
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# 決定境界のプロット
plt.subplot(1, 2, 2)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# グリッドポイントでの予測
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    Z = model(grid_tensor).numpy()
Z = Z.reshape(xx.shape)

# 等高線と散布図
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
plt.scatter(X0[:, 0], X0[:, 1], color='blue', label='Class 0')
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Class 1')

# nn.Linear が学習した重みとバイアスから直線の式を計算して引く
w = model[0].weight.detach().numpy()[0]
b = model[0].bias.detach().numpy()[0]
x_line = np.array([x_min, x_max])
y_line = -(w[0] * x_line + b) / w[1] # w1*x1 + w2*x2 + b = 0 を x2 について解いた式
plt.plot(x_line, y_line, 'k--', label='Boundary Line')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()