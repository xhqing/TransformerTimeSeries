import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def generate_sin_wave(seq_length, num_samples, freq=0.01, noise=0.05):
    X = np.linspace(0, num_samples * seq_length * freq * 2 * np.pi, num_samples * seq_length)
    data = np.sin(X) + noise * np.random.randn(X.shape[0])
    return data.reshape(num_samples, seq_length)


# 数据预处理
seq_length = 50
num_samples = 1000
data = generate_sin_wave(seq_length, num_samples)
train_data = data[:800]
test_data = data[800:]

# 转换为PyTorch张量
train_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(-1)
test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(-1)

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
        return x

# 超参数
input_dim = 1
model_dim = 32
num_heads = 2
num_layers = 2
output_dim = 1
learning_rate = 0.001
num_epochs = 100

# 创建模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = TransformerTimeSeries(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# 训练模型

look_back = 42
look_forward = 8

model.train()
for epoch in range(num_epochs):
    inputs = train_data[:, :look_back, :].to(device)
    targets = train_data[:, -look_forward:, :].to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    import pdb; pdb.set_trace()
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型的状态字典
torch.save(model.state_dict(), "model_checkpoint.pth")


# 预测

# 首先创建一个与之前相同结构的模型
loaded_model = TransformerTimeSeries(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# 加载保存的状态字典
loaded_model.load_state_dict(torch.load("model_checkpoint.pth"))

# 确保模型处于评估模式，这对于包含dropout或batch normalization的模型非常重要
loaded_model.eval()


loaded_model.eval()
with torch.no_grad():
    test_inputs = test_data[:, :look_back, :].to(device)
    test_targets = test_data[:, -look_forward:, :].cpu().numpy()
    test_outputs = loaded_model(test_inputs).cpu().numpy()

import pdb; pdb.set_trace()

# 可视化结果
sample_idx = 1
plt.plot(test_targets[sample_idx], label='True')
plt.plot(test_outputs[sample_idx], label='Predicted')
plt.legend()
plt.show()

