import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 数据准备
y_true = np.array([
    [0,0,1], [0,1,0], [1,0,0], [0,0,1], [1,0,0],
    [0,1,0], [0,1,0], [0,1,0], [0,0,1], [0,1,0]
])
y_score = np.array([
    [0.1, 0.2, 0.7], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8], [0.4, 0.2, 0.4],
    [0.6, 0.3, 0.1], [0.4, 0.2, 0.4], [0.4, 0.1, 0.5], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]
])
n_classes = y_true.shape[1]

# 计算每个类别的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 计算宏观平均
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# 计算微观平均
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 计算加权平均
weights = [2, 5, 3]  # 各类别样本数
weighted_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    weighted_tpr += weights[i] * np.interp(all_fpr, fpr[i], tpr[i])
weighted_tpr /= sum(weights)
fpr["weighted"] = all_fpr
tpr["weighted"] = weighted_tpr
roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

# -------------------------- 分图绘制 --------------------------
# 图1: Class 0 ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr[0], tpr[0], color='blue', label=f'Class 0 (AUC={roc_auc[0]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Class 0')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 图2: Class 1 ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr[1], tpr[1], color='orange', label=f'Class 1 (AUC={roc_auc[1]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Class 1')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 图3: Class 2 ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr[2], tpr[2], color='green', label=f'Class 2 (AUC={roc_auc[2]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Class 2')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 图4: Micro-average ROC  /////////微观平均
plt.figure(figsize=(6, 5))
plt.plot(fpr["micro"], tpr["micro"], color='red', linestyle='-', label=f'Micro-average (AUC={roc_auc["micro"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Micro-average')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 图5: Macro-average ROC  //////宏观平均
plt.figure(figsize=(6, 5))
plt.plot(fpr["macro"], tpr["macro"], color='purple', linestyle='-', label=f'Macro-average (AUC={roc_auc["macro"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Macro-average')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 图6: Weighted-average ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr["weighted"], tpr["weighted"], color='brown', linestyle='-', label=f'Weighted-average (AUC={roc_auc["weighted"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Weighted-average')
plt.legend()
plt.grid(alpha=0.3)
plt.show()