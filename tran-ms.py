# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

def generate_data(sequence_length=10000):
    # 生成正弦和余弦函数数据
    x = np.arange(sequence_length)
    sin_data = np.sin(2*np.pi*x/sequence_length)
    cos_data = np.cos(2*np.pi*x/sequence_length)

    # 组织多元时间序列数据
    data = np.empty((sequence_length, 2))
    data[:,0] = sin_data
    data[:,1] = cos_data

    return data

def create_dataset(sequence, look_back, look_forward):
    data, target = [], []
    for i in range(len(sequence)-look_back-look_forward):
        data.append(sequence[i:i+look_back])
        target.append(sequence[i+look_back:i+look_back+look_forward,0])
    return np.array(data), np.array(target)

# 设置参数
look_back = 140
look_forward = 8
d_model = 64
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dropout = 0.2
input_shape = (look_back, 2)

# 生成数据
seq = generate_data()
X, y = create_dataset(seq, look_back, look_forward)

# 转换为PyTorch张量
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# 构建模型
class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout):
        super(TransformerModel, self).__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout),
            num_encoder_layers)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dropout),
            num_decoder_layers)

        self.linear = nn.Linear(d_model, look_forward)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.linear(output)

        return output

model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
batch_size = 32
num_batches = X.shape[0] // batch_size

for epoch in range(num_epochs):
    for i in range(num_batches):
        batch_x = X_tensor[i*batch_size:(i+1)*batch_size]
        batch_y = y_tensor[i*batch_size:(i+1)*batch_size]

        optimizer.zero_grad()
        outputs = model(batch_x.permute(1, 0, 2), batch_y[:-1].permute(1, 0, 2))
        loss = criterion(outputs.permute(1, 0, 2), batch_y[1:].permute(1, 0, 2))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.5f}")

