import torch
import torch.nn as nn

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 从csv文件中加载训练数据
training_data_df = pd.read_csv("./dataset/sales_data_training.csv", dtype=float)

# 拆分输入和输出
X_training = training_data_df.drop('销售总额', axis=1).values
Y_training = training_data_df[['销售总额']].values

# 加载测试数据
test_data_df = pd.read_csv("./dataset/sales_data_testing.csv", dtype=float)

# 拆分输入输出
X_testing = test_data_df.drop('销售总额', axis=1).values
Y_testing = test_data_df[['销售总额']].values

# 数据归一化：也可以像keras tutorial中那样统一用一个scaler
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# 输入输出都归一化
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# 很重要：训练数据和测试数据必须采用相同的归一化
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# 定义模型超参数
learning_rate = 0.001
training_epochs = 100

# 定义输入输出
number_of_inputs = 9
number_of_outputs = 1

# 定义隐层的节点数：三个隐层
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 构造模型，继承自nn.Module
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, layer_1_nodes)
        self.fc2 = nn.Linear(layer_1_nodes, layer_2_nodes)
        self.fc3 = nn.Linear(layer_2_nodes, layer_3_nodes)
        self.relu = nn.ReLU()

        self.fc4 = nn.Linear(layer_3_nodes, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out


model = NeuralNet(number_of_inputs, number_of_outputs).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):
    # 将numpy转换为pytorch的tensor
    inputs = torch.from_numpy(X_scaled_training).float().to(device)
    labels = torch.from_numpy(Y_scaled_training).float().to(device)

    # 前向计算
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向计算并优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, training_epochs, loss.item()))

with torch.no_grad():
    test_inputs = torch.from_numpy(X_scaled_testing).float().to(device)
    test_labels = torch.from_numpy(Y_scaled_testing).float().to(device)
    outputs = model(test_inputs)
    loss = criterion(outputs, test_labels)

    print('Loss of the network on the test inputs: {}'.format(loss))


torch.save(model, 'model/model.ckpt')
