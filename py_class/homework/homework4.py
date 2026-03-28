import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ====================== 修复中文乱码 ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 加载数据 ======================
df = pd.read_csv(r"E:\python machine_learning\Machine-Learning\py_class\data_set\house_price.csv")

# ====================== 逻辑回归：改为二分类任务 ======================
# 把连续房价 → 二分类标签（高价=1，低价=0，用中位数划分）
X = df.iloc[:, :-1].values
y_price = df.iloc[:, -1].values
# 生成分类标签：房价大于中位数=1，否则=0
y = (y_price > np.median(y_price)).astype(int).reshape(-1, 1)

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 加偏置项（截距项）
def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]


# Sigmoid激活函数（逻辑回归核心）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ==============================================
# 2. 逻辑回归 梯度下降（无正则）
# ==============================================
def logistic_gd(X, y, lr=0.1, epochs=3000):
    X_b = add_bias(X)
    m, n = X_b.shape
    theta = np.random.randn(n, 1) * 0.01
    train_loss = []

    for _ in range(epochs):
        # 线性输出 → Sigmoid转为概率
        z = X_b @ theta
        y_pred = sigmoid(z)

        # 交叉熵损失（逻辑回归专用损失）
        epsilon = 1e-10  # 防止log(0)报错
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        train_loss.append(loss)

        # 梯度计算
        grad = (1 / m) * X_b.T @ (y_pred - y)
        theta -= lr * grad

    return theta, train_loss


theta_gd, loss_gd = logistic_gd(X_train, y_train, lr=1e-5, epochs=3000)

plt.figure(figsize=(7, 5))
plt.plot(loss_gd)
plt.xlabel('迭代次数')
plt.ylabel('交叉熵损失')
plt.title('逻辑回归 梯度下降 训练损失曲线')
plt.grid(True)


# ==============================================
# 3. 逻辑回归 + L2 正则（岭逻辑回归）
# ==============================================
def logistic_ridge_gd(X, y, alpha=1.0, lr=0.1, epochs=3000):
    X_b = add_bias(X)
    m, n = X_b.shape
    theta = np.random.randn(n, 1) * 0.01
    train_loss = []

    for _ in range(epochs):
        z = X_b @ theta
        y_pred = sigmoid(z)

        epsilon = 1e-10
        # 带L2正则的交叉熵损失
        reg_loss = (alpha / (2 * m)) * np.sum(theta[1:] ** 2)
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)) + reg_loss
        train_loss.append(loss)

        # 带正则的梯度
        grad = (1 / m) * X_b.T @ (y_pred - y) + (alpha / m) * np.r_[np.zeros((1, 1)), theta[1:]]
        theta -= lr * grad

    return theta, train_loss


theta_ridge, loss_ridge = logistic_ridge_gd(X_train, y_train, alpha=1.0, lr=1e-5, epochs=3000)

plt.figure(figsize=(7, 5))
plt.plot(loss_ridge)
plt.xlabel('迭代次数')
plt.ylabel('带L2正则的交叉熵损失')
plt.title('L2正则 逻辑回归 损失曲线')
plt.grid(True)

# ==============================================
# 4. 数据归一化 + 逻辑回归（必须归一化！）
# ==============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta_gd_s, loss_gd_s = logistic_gd(X_train_scaled, y_train, lr=0.5, epochs=1500)

plt.figure(figsize=(7, 5))
plt.plot(loss_gd_s)
plt.xlabel('迭代次数')
plt.ylabel('交叉熵损失')
plt.title('归一化后 逻辑回归 损失曲线')
plt.grid(True)


# ==============================================
# 5. 训练 & 测试损失 + 准确率曲线
# ==============================================
def logistic_train_test_loss(X_train, X_test, y_train, y_test, lr=0.5, epochs=800):
    X_train_b = add_bias(X_train)
    X_test_b = add_bias(X_test)
    m, n = X_train_b.shape
    theta = np.random.randn(n, 1) * 0.01

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    epsilon = 1e-10

    for _ in range(epochs):
        # 训练集
        z_tr = X_train_b @ theta
        y_pred_tr = sigmoid(z_tr)
        loss_tr = -np.mean(y_train * np.log(y_pred_tr + epsilon) + (1 - y_train) * np.log(1 - y_pred_tr + epsilon))

        # 测试集
        z_te = X_test_b @ theta
        y_pred_te = sigmoid(z_te)
        loss_te = -np.mean(y_test * np.log(y_pred_te + epsilon) + (1 - y_test) * np.log(1 - y_pred_te + epsilon))

        # 计算准确率
        acc_tr = accuracy_score(y_train, (y_pred_tr > 0.5).astype(int))
        acc_te = accuracy_score(y_test, (y_pred_te > 0.5).astype(int))

        train_loss.append(loss_tr)
        test_loss.append(loss_te)
        train_acc.append(acc_tr)
        test_acc.append(acc_te)

        # 梯度更新
        grad = (1 / m) * X_train_b.T @ (y_pred_tr - y_train)
        theta -= lr * grad

    return train_loss, test_loss, train_acc, test_acc


train_loss, test_loss, train_acc, test_acc = logistic_train_test_loss(
    X_train_scaled, X_test_scaled, y_train, y_test, lr=0.5, epochs=800
)

# 绘制损失曲线
plt.figure(figsize=(7, 5))
plt.plot(train_loss, label='训练损失')
plt.plot(test_loss, label='测试损失')
plt.xlabel('迭代次数')
plt.ylabel('交叉熵损失')
plt.title('逻辑回归 训练&测试损失')
plt.legend()
plt.grid(True)

# 绘制准确率曲线
plt.figure(figsize=(7, 5))
plt.plot(train_acc, label='训练准确率')
plt.plot(test_acc, label='测试准确率')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.title('逻辑回归 训练&测试准确率')
plt.legend()
plt.grid(True)

# ====================== 最终模型评估 ======================
X_test_b = add_bias(X_test_scaled)
y_pred_prob = sigmoid(X_test_b @ theta_gd_s)
y_pred = (y_pred_prob > 0.5).astype(int)

print("=" * 50)
print("逻辑回归 测试集评估")
print("=" * 50)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['低价房', '高价房']))

plt.show()

#总结:学习率过大会导致出现剧烈抖动（0.1）
#     学习率调到1e-5曲线平滑