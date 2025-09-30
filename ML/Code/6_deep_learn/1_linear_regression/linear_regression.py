import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
    1. 生成训练数据
"""

torch.manual_seed(42)  # 设置随机种子，保证每次生成的随机数相同
X = torch.randn(100, 2)  # 100 个样本，每个样本两个特征
true_w = torch.tensor([2.0, 3.0])  # 真实的权重
true_b = 4.0  # 真实的偏置
Y = X @ true_w + true_b + torch.randn(100) * 0.5

"""
    2. 定义线性回归模型
"""


class LinearRegressionModel(nn.Module):
    def __init__(self):
        # 初始化父类 nn.Module
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层，输入为2个特征，输出为1个预测值
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        # 前向传播，返回预测值
        return self.linear(x)


# 创建模型实例
model = LinearRegressionModel()

"""
    3. 定义损失函数和优化器
"""

criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器


"""
    4. 训练模型
"""


num_epochs = 1000  # 训练 1000 轮
for epoch in range(num_epochs):

    # 前向传播
    predictions = model(X)  # 模型输出预测值
    loss = criterion(predictions.squeeze(), Y)  # 计算损失

    # 反向传播
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新模型参数

    # 打印损失
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}")

"""
    5. 模型评估
"""
print(f"Predicted weight: {model.linear.weight.data.numpy()}")
print(f"Predicted bias: {model.linear.bias.data.numpy()}")

with torch.no_grad():  # 在评估时不计算梯度
    predictions = model(X)  # 模型输出预测值
    loss = criterion(predictions.squeeze(), Y)  # 计算损失
    print(f"Test Loss: {loss.item():.4f}")

# 可视化预测与实际值
plt.scatter(X[:, 0], Y, color="blue", label="True values")
plt.scatter(X[:, 0], predictions, color="red", label="Predictions")
plt.legend()
plt.show()
