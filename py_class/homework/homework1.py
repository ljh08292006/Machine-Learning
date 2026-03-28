import matplotlib.pyplot as plt

# 1. 准备数据：(真实标签Y_true, 预测概率Y_score)
data = [
    (1, 0.90), (1, 0.42), (0, 0.20), (1, 0.60), (0, 0.50),
    (0, 0.40), (1, 0.70), (1, 0.40), (0, 0.65), (0, 0.35)
]

# 2. 按预测概率降序排序（核心步骤：不同阈值的基础）
sorted_data = sorted(data, key=lambda x: -x[1])

# 3. 统计正负样本总数
P = sum(1 for y, _ in data if y == 1)  # 正样本总数：4
N = sum(1 for y, _ in data if y == 0)  # 负样本总数：6

# 4. 初始化指标变量和点列表
# PR曲线点：(Recall, Precision)，起点(0,1)（阈值无穷大时，无样本被预测为正）
pr_points = [(0.0, 1.0)]
# ROC曲线点：(FPR, TPR)，起点(0,0)（阈值无穷大时，无样本被预测为正）
roc_points = [(0.0, 0.0)]
TP = 0  # 真阳性数
FP = 0  # 假阳性数

# 5. 遍历排序后的样本，计算每个阈值下的指标
for y_true, _ in sorted_data:
    if y_true == 1:
        TP += 1  # 预测为正且真实为正
    else:
        FP += 1  # 预测为正但真实为负

    # 计算PR曲线的Recall和Precision
    recall = TP / P  # 召回率 = 正确预测的正样本 / 所有正样本
    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0  # 精确率（避免除以0）
    pr_points.append((recall, precision))

    # 计算ROC曲线的TPR和FPR
    tpr = TP / P  # 真阳性率 = Recall
    fpr = FP / N  # 假阳性率 = 错误预测的负样本 / 所有负样本
    roc_points.append((fpr, tpr))

# ========== 差异点1：PR曲线终点强制补充 ==========
# sklearn的precision_recall_curve不会强制添加(1.0, 0.0)终点，
# 而是根据实际阈值计算到最后一个样本（Recall=1时Precision为真实值，而非0）
pr_points.append((1.0, 0.0))

# 6. 计算AUC（ROC曲线下面积，梯形法）
auc = 0.0
for i in range(1, len(roc_points)):
    x0, y0 = roc_points[i - 1]
    x1, y1 = roc_points[i]
    # ========== 差异点2：ROC-AUC计算的样本点数量 ==========
    # 手动实现遍历了所有排序后的样本（10个）+ 初始点，共11个ROC点；
    # 而sklearn的roc_curve会自动去重（相同预测概率的样本合并为同一个阈值），
    # 导致计算AUC的样本点更少，最终AUC值不同
    auc += (x1 - x0) * (y0 + y1) / 2

# 7. 绘制图表（PR曲线 + ROC曲线）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.figure(figsize=(12, 5))  # 设置画布大小

# 子图1：PR曲线
plt.subplot(1, 2, 1)
recalls, precisions = zip(*pr_points)
plt.plot(recalls, precisions, marker='o', color='#1f77b4', linewidth=2, markersize=6)
plt.xlabel('召回率 (Recall)', fontsize=10)
plt.ylabel('精确率 (Precision)', fontsize=10)
plt.title('PR曲线', fontsize=12)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)

# 子图2：ROC曲线
plt.subplot(1, 2, 2)
fprs, tprs = zip(*roc_points)
plt.plot(fprs, tprs, marker='o', color='#ff7f0e', linewidth=2, markersize=6, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='随机猜测')  # 对角线
plt.xlabel('假阳性率 (FPR)', fontsize=10)
plt.ylabel('真阳性率 (TPR)', fontsize=10)
plt.title('ROC曲线', fontsize=12)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()  # 调整子图间距
plt.show()

# 打印关键结果
print(f"正样本数(P)：{P}，负样本数(N)：{N}")
print(f"ROC曲线的AUC值：{auc:.4f}")

# ========== 差异点3：PR-AUC计算缺失 ==========
# 手动实现仅计算了ROC-AUC，未计算PR-AUC；
# 而sklearn版本同时计算了ROC-AUC和PR-AUC，且PR曲线的点生成逻辑不同：
# sklearn的precision_recall_curve会：
# 1. 自动生成所有唯一的预测概率作为阈值；
# 2. 对每个阈值计算Precision和Recall；
# 3. 最终返回的Precision数组会比Recall少一个元素（无强制终点）；
# 4. PR-AUC计算基于实际阈值的Precision/Recall，而非手动补充的(1,0)