import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成多元正弦波数据
def generate_multi_sin_wave(seq_length, num_samples, num_features, freq=0.01, noise=0.05):
    X = np.linspace(0, num_samples * seq_length * freq * 2 * np.pi, num_samples * seq_length)
    data = np.array([np.sin(X + i * np.pi / num_features) + noise * np.random.randn(X.shape[0]) for i in range(num_features)]).T
    return data.reshape(num_samples, seq_length, num_features)

# 数据预处理
seq_length = 50
num_samples = 1000
num_features = 3
data = generate_multi_sin_wave(seq_length, num_samples, num_features)
train_data = data[:800]
test_data = data[800:]

# 转换为PyTorch张量
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)

# 构建Transformer模型
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.transformer = nn.Transformer(model_dim, num_heads, num_layers)
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.output_fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer(x, x)
        x = self.output_fc(x)
        x = x.transpose(0, 1)
        import pdb; pdb.set_trace()
        return x

# 超参数
input_dim = num_features
model_dim = 32
num_heads = 2
num_layers = 2
output_dim = 1
learning_rate = 0.001
num_epochs = 100

# 创建模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerTimeSeries(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# 训练模型
model.train()
for epoch in range(num_epochs):
    inputs = train_data[:, :-8, :].to(device)
    targets = train_data[:, -8:, 0].unsqueeze(-1).to(device)  # 修改目标以预测未来8个点

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测
model.eval()
with torch.no_grad():
    test_inputs = test_data[:, :-8, :].to(device)
    test_targets = test_data[:, -8:, 0].unsqueeze(-1).cpu().numpy()  # 修改目标以预测未来8个点
    test_outputs = model(test_inputs).cpu().numpy()

# 将预测结果与实际结果进行比较
plt.plot(test_outputs[0, :, 0], label='Predicted')
plt.plot(test_targets[0, :, 0], label='Actual')
plt.legend()
plt.show()

