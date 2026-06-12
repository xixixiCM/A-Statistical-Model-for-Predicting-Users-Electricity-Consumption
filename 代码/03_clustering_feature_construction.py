import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.stats import kurtosis, skew, levene
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import os
out_dir = "stat_describe"
os.makedirs(out_dir, exist_ok=True)
#1、加载归一化日平均曲线数据
file_path = "daily_profile_norm.csv"
df = pd.read_csv(file_path, index_col=0, encoding='utf-8-sig')
print(f"数据形状: {df.shape}")
X = df.values

#2、PCA：评估维度与有效秩
pca = PCA()
X_centered = X - X.mean(axis=0)
pca.fit(X_centered)
cum_var = np.cumsum(pca.explained_variance_ratio_)
#确定达到95%方差所需的主成分数
n_comp_80 = np.argmax(cum_var >= 0.95) + 1
print(f"\n===== PCA 维度分析 =====")
print(f"前5个主成分累计方差贡献率: {cum_var[:5]}")
print(f"达到95%方差需要的主成分数: {n_comp_80}")
print(f"前10主成分累计方差: {cum_var[9]:.4f}")

#有效秩：奇异值平方和占比 > 0.01 的个数
svd_vals = np.linalg.svd(X_centered, compute_uv=False)
effective_rank = np.sum(svd_vals**2 / np.sum(svd_vals**2) > 0.01)
print(f"有效秩 (奇异值贡献>1%): {effective_rank}")

#绘图：累计方差曲线图
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cum_var)+1), cum_var, 'b-o', markersize=3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95%方差阈值')
plt.title('PCA累计方差贡献率')
plt.xlabel('主成分序号')
plt.ylabel('累计方差比例')
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/pca_cumulative_variance.png", dpi=150)
plt.close()

#3、霍普金斯统计量（分析聚类趋势）
def hopkins_statistic(X, sample_ratio=0.1):
    n = X.shape[0]
    m = int(sample_ratio * n)
    # 随机选取m个真实点
    rand_idx = np.random.choice(n, m, replace=False)
    # 在特征空间内生成 m 个均匀随机点
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    X_random = np.random.uniform(min_vals, max_vals, (m, X.shape[1]))
    # 计算每个随机点到最近真实点的距离
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    u_dist, _ = nbrs.kneighbors(X_random, n_neighbors=2)  # 随机点
    w_dist, _ = nbrs.kneighbors(X[rand_idx], n_neighbors=2) # 真实点
    u_sum = np.sum(u_dist[:, 1])
    w_sum = np.sum(w_dist[:, 1])
    H = u_sum / (u_sum + w_sum)
    return H
np.random.seed(42)
H = hopkins_statistic(X)
print(f"\n===== 聚类趋势 =====")
print(f"霍普金斯统计量: {H:.4f}")
if H >0.8:
    print("->数据具有显著聚类结构，聚类有意义")
elif H > 0.5:
    print("->中等聚类趋势")
else:
    print("->数据接近均匀分布，聚类可能无意义")

#4、Levene检验：各时段方差是否齐性
groups = [X[:, i] for i in range(X.shape[1])]
stat, p_levene = levene(*groups)
print(f"\n===== 方差齐性检验 =====")
print(f"Levene检验 p值: {p_levene:.6f}")
if p_levene < 0.05:
    print("   → 各时段方差不相等 → 违背K-Means的方差齐性假设")
else:
    print("   → 方差齐性，K-Means假设可接受")

#5、汇总诊断报表（保存到文本）
with open(f"{out_dir}/stat_diagnosis_summary.txt", "w", encoding="utf-8") as f:
    f.write("===== 聚类前统计诊断报告 =====\n")
    f.write(f"数据形状: {X.shape}\n")
    f.write(f"PCA达到95%方差的主成分数: {n_comp_80}\n")
    f.write(f"有效秩: {effective_rank}\n")
    f.write(f"霍普金斯统计量: {H:.4f}\n")
    f.write(f"Levene检验 p值: {p_levene:.6f}\n")
    if n_comp_80 > 20 or effective_rank > 30:
        f.write(" - 维度较高，需先降维处理或使用深度聚类方法\n")
    if H > 0.8:
        f.write(" - 聚类趋势强，值得进行聚类分析\n")
print(f"\n✅ 统计描述完成。所有结果和图表已保存至 '{out_dir}' 文件夹")