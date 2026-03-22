import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# ===================== 1. 题目原始正确数据 =====================
# 真实标签（one-hot格式，对应10个样本）
y_true = np.array([
    [0, 0, 1],  # 样本1
    [0, 1, 0],  # 样本2
    [1, 0, 0],  # 样本3
    [0, 0, 1],  # 样本4
    [1, 0, 0],  # 样本5
    [0, 1, 0],  # 样本6
    [0, 1, 0],  # 样本7
    [0, 1, 0],  # 样本8
    [0, 0, 1],  # 样本9
    [0, 1, 0]   # 样本10
])

# 模型预测概率分数（对应10个样本）
y_score = np.array([
    [0.1, 0.2, 0.7],  # 样本1
    [0.1, 0.6, 0.3],  # 样本2
    [0.5, 0.2, 0.3],  # 样本3
    [0.1, 0.1, 0.8],  # 样本4
    [0.4, 0.2, 0.4],  # 样本5
    [0.6, 0.3, 0.1],  # 样本6
    [0.4, 0.2, 0.4],  # 样本7
    [0.4, 0.1, 0.5],  # 样本8
    [0.1, 0.1, 0.8],  # 样本9
    [0.1, 0.8, 0.1]   # 样本10
])

# 类别数量和名称
n_classes = y_true.shape[1]
class_names = ['类别0', '类别1', '类别2']
# 曲线颜色（对应3个类别）
colors = ['aqua', 'darkorange', 'cornflowerblue']

# ===================== 2. 计算每个类别的ROC和AUC =====================
fpr = dict()
tpr = dict()
roc_auc = dict()

# 逐个类别计算FPR（假阳性率）、TPR（真阳性率）、AUC
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ===================== 3. 计算Micro平均ROC（整体性能） =====================
fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# ===================== 4. 绘制四张独立的图（2行2列布局） =====================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # 扁平化子图数组，方便按顺序绘制

# --- 第一张图：类别0的ROC曲线 ---
axes[0].plot(fpr[0], tpr[0], color=colors[0], lw=2,
             label=f'{class_names[0]} (AUC = {roc_auc[0]:.2f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测 (AUC=0.5)')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('假阳性率 (FPR)', fontsize=10)
axes[0].set_ylabel('真阳性率 (TPR)', fontsize=10)
axes[0].set_title(f'{class_names[0]} ROC曲线', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# --- 第二张图：类别1的ROC曲线 ---
axes[1].plot(fpr[1], tpr[1], color=colors[1], lw=2,
             label=f'{class_names[1]} (AUC = {roc_auc[1]:.2f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测 (AUC=0.5)')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('假阳性率 (FPR)', fontsize=10)
axes[1].set_ylabel('真阳性率 (TPR)', fontsize=10)
axes[1].set_title(f'{class_names[1]} ROC曲线', fontsize=12)
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

# --- 第三张图：类别2的ROC曲线 ---
axes[2].plot(fpr[2], tpr[2], color=colors[2], lw=2,
             label=f'{class_names[2]} (AUC = {roc_auc[2]:.2f})')
axes[2].plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测 (AUC=0.5)')
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('假阳性率 (FPR)', fontsize=10)
axes[2].set_ylabel('真阳性率 (TPR)', fontsize=10)
axes[2].set_title(f'{class_names[2]} ROC曲线', fontsize=12)
axes[2].legend(loc='lower right')
axes[2].grid(alpha=0.3)

# --- 第四张图：Micro平均ROC曲线（整体性能） ---
axes[3].plot(fpr_micro, tpr_micro, color='red', lw=2, linestyle='--',
             label=f'Micro平均 (AUC = {roc_auc_micro:.2f})')
axes[3].plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测 (AUC=0.5)')
axes[3].set_xlim([0.0, 1.0])
axes[3].set_ylim([0.0, 1.05])
axes[3].set_xlabel('假阳性率 (FPR)', fontsize=10)
axes[3].set_ylabel('真阳性率 (TPR)', fontsize=10)
axes[3].set_title('Micro平均ROC曲线（整体性能）', fontsize=12)
axes[3].legend(loc='lower right')
axes[3].grid(alpha=0.3)

# 调整子图间距，避免标签重叠
plt.tight_layout()
plt.show()