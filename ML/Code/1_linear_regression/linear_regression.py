import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 准备数据（这里使用随机生成的数据作为示例）
np.random.seed(42)
n_samples = 100
n_features = 3

# 生成特征数据
X = np.random.randn(n_samples, n_features)

# 生成目标变量（添加一些噪声）
true_coef = np.array([3.5, 2.0, -1.2])
y = X.dot(true_coef) + np.random.randn(n_samples) * 0.5 + 1.0

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 进行预测
y_pred = model.predict(X_test)

# 5. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"模型系数: {model.coef_}")
print(f"模型截距: {model.intercept_}")
print(f"均方误差(MSE): {mse:.4f}")
print(f"决定系数(R²): {r2:.4f}")