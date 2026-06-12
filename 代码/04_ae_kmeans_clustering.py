import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

#1、参数设置
INPUT_FILE = "daily_profile_norm.csv"
BEST_K = 4
LATENT_DIM = 8
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "ae_kmeans_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2、固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

#3、读取数据
print("1. 读取输入数据")
df = pd.read_csv(INPUT_FILE, index_col=0, encoding="utf-8-sig")
X = df.values.astype(np.float32)   # shape: (n_users, 96)
user_ids = df.index.tolist()
if not np.isfinite(X).all():
    raise ValueError("输入数据包含 NaN 或无穷值，请先清洗。")
print(f"输入文件: {INPUT_FILE}")
print(f"数据形状: {X.shape}")   # (用户数, 96)
#4、定义 Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=96, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()   # 因为输入是归一化后的 0~1 曲线
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
#5、构建 DataLoader
tensor_x = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(tensor_x)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#6、训练 Autoencoder
print("\n" + "=" * 60)
print("2. 训练 Autoencoder")
print("=" * 60)
model = AutoEncoder(input_dim=X.shape[1], latent_dim=LATENT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for (batch_x,) in loader:
        batch_x = batch_x.to(DEVICE)
        optimizer.zero_grad()
        recon, _ = model(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    epoch_loss /= len(dataset)
    loss_history.append(epoch_loss)
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch [{epoch:03d}/{EPOCHS}]  Reconstruction Loss = {epoch_loss:.6f}")

#7、AE重构曲线损失loss曲线
plt.figure(figsize=(8, 5))
plt.plot(loss_history, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("Autoencoder Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_training_loss.png"), dpi=150)
plt.close()

#8、提取 latent 表示
print("\n" + "=" * 60)
print("3. 提取 latent 表示")
print("=" * 60)
model.eval()
with torch.no_grad():
    _, Z = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    Z = Z.cpu().numpy()
print(f"latent 形状: {Z.shape}")   # (n_users, latent_dim)
latent_df = pd.DataFrame(Z, index=user_ids, columns=[f"latent_{i}" for i in range(LATENT_DIM)])
latent_df.to_csv(os.path.join(OUTPUT_DIR, "ae_latent_features.csv"), encoding="utf-8-sig")
print("已保存: ae_latent_features.csv")

#9、手肘法图绘制
print("\n" + "=" * 60)
print("绘制手肘法图")
print("=" * 60)
K_range = range(2, 16)
inertias = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    km.fit(Z)
    inertias.append(km.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=6)
plt.xlabel("K 值")
plt.ylabel("簇内误差平方和")
plt.title("K-Means 拐点图")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elbow_method.png"), dpi=150)
plt.close()
print("✅ 手肘拐点图已保存：elbow_method.png")

# 10、原始特征直接 KMeans（基线）
print("\n" + "=" * 60)
print("4. 原始 96维特征直接 KMeans（基线）")
print("=" * 60)
kmeans_raw = KMeans(n_clusters=BEST_K, random_state=RANDOM_SEED, n_init=20)
labels_raw = kmeans_raw.fit_predict(X)
sil_raw = silhouette_score(X, labels_raw)
ch_raw = calinski_harabasz_score(X, labels_raw)
db_raw = davies_bouldin_score(X, labels_raw)
print(f"[Raw + KMeans] silhouette = {sil_raw:.4f}")
print(f"[Raw + KMeans] calinski_harabasz = {ch_raw:.4f}")
print(f"[Raw + KMeans] davies_bouldin = {db_raw:.4f}")

#11、多种聚类方法在 AE的latent上对比，并保存对比结果
print("\n" + "=" * 60)
print("5. AE 特征 + 多种聚类方法对比实验")
print("=" * 60)
results = []
# 方法1：AE + KMeans
labels_kmeans = KMeans(n_clusters=BEST_K, random_state=RANDOM_SEED, n_init=20).fit_predict(Z)
sil_kmeans = silhouette_score(Z, labels_kmeans)
ch_kmeans = calinski_harabasz_score(Z, labels_kmeans)
db_kmeans = davies_bouldin_score(Z, labels_kmeans)
results.append(["AE + KMeans", sil_kmeans, ch_kmeans, db_kmeans])
print(f"[AE + KMeans] sil={sil_kmeans:.4f}, CH={ch_kmeans:.4f}, DB={db_kmeans:.4f}")
# 方法2：AE + 层次聚类
labels_agg = AgglomerativeClustering(n_clusters=BEST_K).fit_predict(Z)
sil_agg = silhouette_score(Z, labels_agg)
ch_agg = calinski_harabasz_score(Z, labels_agg)
db_agg = davies_bouldin_score(Z, labels_agg)
results.append(["AE + Hierarchical", sil_agg, ch_agg, db_agg])
print(f"[AE + 层次聚类] sil={sil_agg:.4f}, CH={ch_agg:.4f}, DB={db_agg:.4f}")
# 方法3：AE + GMM 高斯混合
gmm = GaussianMixture(n_components=BEST_K, random_state=RANDOM_SEED)
labels_gmm = gmm.fit_predict(Z)
sil_gmm = silhouette_score(Z, labels_gmm)
ch_gmm = calinski_harabasz_score(Z, labels_gmm)
db_gmm = davies_bouldin_score(Z, labels_gmm)
results.append(["AE + GMM", sil_gmm, ch_gmm, db_gmm])
print(f"[AE + GMM]     sil={sil_gmm:.4f}, CH={ch_gmm:.4f}, DB={db_gmm:.4f}")
metrics_df = pd.DataFrame(results, columns=["方法", "silhouette", "calinski_harabasz", "davies_bouldin"])
metrics_path = os.path.join(OUTPUT_DIR, "ae_multi_clustering_metrics.csv")
metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
print("\n===== 聚类方法对比结果 =====")
print(metrics_df.round(4))

# 12. 保存最终使用的聚类（KMeans）
labels_ae = labels_kmeans
cluster_result_df = df.copy()
cluster_result_df["cluster"] = labels_ae
cluster_file = os.path.join(OUTPUT_DIR, "daily_profile_norm_with_cluster_ae.csv")
cluster_result_df.to_csv(cluster_file, encoding="utf-8-sig")
print(f"\n聚类标签文件已保存: {cluster_file}")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 13、可视化图绘制
#①聚类效果
print("\n" + "=" * 60)
print("6. PCA 可视化")
print("=" * 60)
pca = PCA(n_components=2, random_state=RANDOM_SEED)
Z_pca = pca.fit_transform(Z)
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10(np.linspace(0, 1, BEST_K))
for c in range(BEST_K):
    mask = labels_ae == c
    plt.scatter(
        Z_pca[mask, 0], Z_pca[mask, 1],
        label=f"Cluster {c}",
        s=35, alpha=0.75, c=[colors[c]]
    )
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.title(f"AE Latent + KMeans (k={BEST_K})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_kmeans_pca.png"), dpi=150)
plt.close()
#②簇中心曲线
print("\n" + "=" * 60)
print("7. 各类中心曲线可视化")
print("=" * 60)
slots = np.arange(X.shape[1])
plt.figure(figsize=(10, 6))
for c in range(BEST_K):
    center_curve = X[labels_ae == c].mean(axis=0)
    plt.plot(slots, center_curve, linewidth=2, label=f"Cluster {c}")
plt.xlabel("15分钟时段 (0-95)")
plt.ylabel("归一化负荷")
plt.title("AE + KMeans 各类中心曲线")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ae_cluster_centers.png"), dpi=150)
plt.close()

print("\n" + "=" * 60)
print("8. 各类样本数量")
print("=" * 60)
for c in range(BEST_K):
    cnt = np.sum(labels_ae == c)
    print(f"Cluster {c}: {cnt} 个用户")
print("\n✅ 全部完成。输出目录：", OUTPUT_DIR)