import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ====================== 修复中文乱码 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows自带黑体
plt.rcParams['axes.unicode_minus'] = False

# ====================== 修复路径转义错误 ======================
df = pd.read_csv(r"E:\python machine_learning\Machine-Learning\py_class\data_set\house_price.csv")

# 假设最后一列是房价
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 加偏置项
def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

# ==============================================
# 1. 最小二乘法
# ==============================================
X_train_b = add_bias(X_train)
theta_ls = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

y_pred_train_ls = X_train_b @ theta_ls
y_pred_test_ls = add_bias(X_test) @ theta_ls

plt.figure(figsize=(7,5))
plt.scatter(y_train, y_pred_train_ls, s=5, alpha=0.5, label='训练集')
plt.scatter(y_test, y_pred_test_ls, s=5, alpha=0.5, label='测试集')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('最小二乘法 拟合结果')
plt.legend()
plt.grid(True)

# ==============================================
# 2. 梯度下降（无正则）
# ==============================================
def gradient_descent(X, y, lr=1e-5, epochs=3000):
    X_b = add_bias(X)
    m, n = X_b.shape
    theta = np.random.randn(n, 1) * 0.01
    train_loss = []
    for _ in range(epochs):
        y_pred = X_b @ theta
        loss = mean_squared_error(y, y_pred)
        train_loss.append(loss)
        grad = (2 / m) * X_b.T @ (y_pred - y)
        theta -= lr * grad
    return theta, train_loss

theta_gd, loss_gd = gradient_descent(X_train, y_train, lr=0.00001, epochs=3000)

plt.figure(figsize=(7,5))
plt.plot(loss_gd)
plt.xlabel('迭代次数')
plt.ylabel('MSE 损失')
plt.title('梯度下降 训练损失曲线')
plt.grid(True)

# ==============================================
# 3. 梯度下降 + L2 正则
# ==============================================
def ridge_gd(X, y, alpha=10, lr=1e-5, epochs=3000):
    X_b = add_bias(X)
    m, n = X_b.shape
    theta = np.random.randn(n, 1) * 0.01
    train_loss = []
    for _ in range(epochs):
        y_pred = X_b @ theta
        reg = (alpha / m) * np.sum(theta[1:] ** 2)
        loss = mean_squared_error(y, y_pred) + reg
        train_loss.append(loss)
        grad = (2/m)*X_b.T@(y_pred - y) + (2*alpha/m)*np.r_[np.zeros((1,1)), theta[1:]]
        theta -= lr * grad
    return theta, train_loss

theta_ridge, loss_ridge = ridge_gd(X_train, y_train, alpha=10, lr=0.00001, epochs=3000)

plt.figure(figsize=(7,5))
plt.plot(loss_ridge)
plt.xlabel('迭代次数')
plt.ylabel('带L2正则的损失')
plt.title('L2正则梯度下降 损失曲线')
plt.grid(True)

# ==============================================
# 4. 数据归一化 + 梯度下降
# ==============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta_gd_s, loss_gd_s = gradient_descent(X_train_scaled, y_train, lr=0.01, epochs=1500)

plt.figure(figsize=(7,5))
plt.plot(loss_gd_s)
plt.xlabel('迭代次数')
plt.ylabel('MSE 损失')
plt.title('归一化后 梯度下降损失曲线')
plt.grid(True)

# ==============================================
# 5. 训练 & 测试损失曲线
# ==============================================
def get_train_test_loss(X_train, X_test, y_train, y_test, lr=0.01, epochs=800):
    X_train_b = add_bias(X_train)
    X_test_b = add_bias(X_test)
    m, n = X_train_b.shape
    theta = np.random.randn(n, 1) * 0.01
    train_loss = []
    test_loss = []
    for _ in range(epochs):
        y_pred_tr = X_train_b @ theta
        y_pred_te = X_test_b @ theta
        train_loss.append(mean_squared_error(y_train, y_pred_tr))
        test_loss.append(mean_squared_error(y_test, y_pred_te))
        grad = (2/m)*X_train_b.T @ (y_pred_tr - y_train)
        theta -= lr * grad
    return train_loss, test_loss

train_loss, test_loss = get_train_test_loss(
    X_train_scaled, X_test_scaled, y_train, y_test, lr=0.01, epochs=800
)

plt.figure(figsize=(7,5))
plt.plot(train_loss, label='训练损失')
plt.plot(test_loss, label='测试损失')
plt.xlabel('迭代次数')
plt.ylabel('MSE')
plt.title('训练 & 测试损失曲线')
plt.legend()
plt.grid(True)

plt.show()

#总结：梯度下降算法学习率低（1e-5），且迭代次数也很多（3000）左右
#     归一化后学习率大幅提高（0.01）,迭代次数也大幅下降到1400左右
#  I2正则化则防止模型过拟合
#学习率过大会使曲线疯狂抖动→ 梯度爆炸 → 模型无法收敛，最后甚至报错崩溃（学习率设为0，01的结果）
#加入正则化可以防止模型过拟合
#学习率太大 → 模型训练不收敛
#训练集非线性（+ 模型收敛 + 模型复杂度偏高） → 过拟合