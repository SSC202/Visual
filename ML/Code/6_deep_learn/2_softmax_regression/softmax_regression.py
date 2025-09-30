import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device\n")

"""
    1. 读取 Fashion-MNIST 数据集
"""


# 训练集
training_data = datasets.FashionMNIST(
    root="./data",  # 存储训练数据的路径
    train=True,  # 训练集
    download=True,  # 从网络下载
    transform=ToTensor(),
)  # 将图片转换为张量

# 测试集
test_data = datasets.FashionMNIST(
    root="./data",  # 存储测试数据的路径
    train=False,  # 测试集
    download=True,  # 从网络下载
    transform=ToTensor(),
)  # 将图片转换为张量

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


"""
    2. 定义 softmax 回归模型
"""


class SoftmaxRegression(nn.Module):
    def __init__(self):
        # 初始化父类 nn.Module
        super(SoftmaxRegression, self).__init__()
        # 定义一个扁平层，在线性层之前调整输入形状
        self.flatten = nn.Flatten()
        # 定义一个线性层，输入为 28*28 个特征，输出为 10 个类别
        self.linear = nn.Linear(28 * 28, 10)
        # 定义激活层，使用 Softmax 激活函数
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 前向传播
        x = self.flatten(x)  # 扁平化输入
        x = self.linear(x)  # 线性变换
        x = self.softmax(x)  # Softmax 激活
        return x


# 创建模型实例
model = SoftmaxRegression().to(device)

"""
    3. 定义损失函数和优化器
"""

# 超参数
learning_rate = 0.1
batch_size = 256
epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""
    4. 训练模型
"""

train_losses = []  # 记录每个 epoch 的训练损失
train_accuracies = []  # 记录每个 epoch 的训练准确率

for epoch in range(epochs):

    model.train()  # 设置模型为训练模式

    running_loss = 0.0
    correct = 0
    total = 0

    # 小批量梯度下降
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # 前向传播
        predictions = model(data)  # 模型输出预测值
        loss = criterion(predictions, target)  # 计算损失

        # 反向传播
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

        # 统计数据
        running_loss += loss.item()  # 累加损失
        _, predicted = torch.max(predictions.data, 1)  # 获取预测类别
        total += target.size(0)  # 累加样本数
        correct += (predicted == target).sum().item()  # 累加正确预测数

    # 计算每个epoch的损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    if (epoch + 1) % 1 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
        )

"""
    5. 模型评估
"""
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

test_accuracy = 100 * correct / total
print(f"测试集准确率: {test_accuracy:.2f}%")

"""
    6. 可视化训练过程
"""
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()
