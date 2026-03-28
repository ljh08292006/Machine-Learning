import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

# 1. 准备数据，和手动版本完全一致
# 真实标签 Y_true
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0])
# 预测概率 Y_score
y_score = np.array([0.90, 0.42, 0.20, 0.60, 0.50, 0.40, 0.70, 0.40, 0.65, 0.35])

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 计算PR曲线相关指标
# ========== 对比手动实现：sklearn的precision_recall_curve逻辑 ==========
# 1. 自动提取y_score中所有唯一值作为阈值（去重），避免重复计算相同阈值；
# 2. 对每个阈值，计算“预测为正的样本”（y_score >= 阈值）的Precision和Recall；
# 3. 最终返回的precision数组长度 = recall数组长度 + 1（因最后一个阈值为负无穷时，Precision=0）；
# 4. 不会强制添加(1.0, 0.0)终点，而是保留真实计算结果
precision, recall, _ = precision_recall_curve(y_true, y_score)
# 计算PR曲线下的面积（PR-AUC）：基于真实的Precision/Recall点，而非手动补充的点
pr_auc = auc(recall, precision)

# 3. 计算ROC曲线相关指标
# ========== 对比手动实现：sklearn的roc_curve逻辑 ==========
# 1. 同样提取y_score中唯一值作为阈值（去重），减少计算点；
# 2. 对每个阈值计算FPR和TPR，避免重复样本导致的多余点；
# 3. 最终AUC计算基于去重后的点，与手动实现的“全样本遍历”结果不同
fpr, tpr, _ = roc_curve(y_true, y_score)
# 计算ROC曲线下的面积（ROC-AUC）：基于去重后的阈值点
roc_auc = auc(fpr, tpr)

# 4. 绘制图表（PR曲线 + ROC曲线）
plt.figure(figsize=(12, 5))

# 子图1：PR曲线
plt.subplot(1, 2, 1)
# 使用sklearn的可视化工具快速绘制：基于真实计算的Precision/Recall
PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=plt.gca(), color='#1f77b4')
plt.title(f'PR曲线 (PR-AUC = {pr_auc:.4f})', fontsize=12)
plt.xlabel('召回率 (Recall)', fontsize=10)
plt.ylabel('精确率 (Precision)', fontsize=10)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)

# 子图2：ROC曲线
plt.subplot(1, 2, 2)
# 使用sklearn的可视化工具快速绘制：基于去重后的FPR/TPR
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=plt.gca(), color='#ff7f0e')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测')
plt.title(f'ROC曲线 (ROC-AUC = {roc_auc:.4f})', fontsize=12)
plt.xlabel('假阳性率 (FPR)', fontsize=10)
plt.ylabel('真阳性率 (TPR)', fontsize=10)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 打印关键结果
print(f"ROC-AUC值（sklearn计算）：{roc_auc:.4f}")
print(f"PR-AUC值（sklearn计算）：{pr_auc:.4f}")