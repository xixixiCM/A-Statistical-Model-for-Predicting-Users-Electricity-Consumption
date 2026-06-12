import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

#1、加载数据
cluster_file = "ae_kmeans_results/daily_profile_norm_with_cluster_ae.csv"
df_norm = pd.read_csv(cluster_file, index_col=0, encoding='utf-8-sig')
labels = df_norm['cluster'].values
K = len(df_norm['cluster'].unique())
print(f"聚类类别数: {K}")
print("各类别样本数:")
print(df_norm['cluster'].value_counts().sort_index())
latent_file = "ae_kmeans_results/ae_latent_features.csv"
if not os.path.exists(latent_file):
    raise FileNotFoundError(f"未找到 {latent_file}，请先运行 AE 聚类脚本生成潜在特征文件。")
latent_df = pd.read_csv(latent_file, index_col=0, encoding='utf-8-sig')
#确保用户顺序一致
common_idx = df_norm.index.intersection(latent_df.index)
df_norm = df_norm.loc[common_idx]
latent_df = latent_df.loc[common_idx]
labels = df_norm['cluster'].values
Z = latent_df.values  # (n_users, latent_dim)

# 原始负荷数据（用于量级和季节性统计）
load_data = pd.read_csv("LD2011_2014_cleaned.csv", index_col=0, parse_dates=True)
common_users = list(set(load_data.columns) & set(df_norm.index))
load_data = load_data[common_users]
df_norm = df_norm.loc[common_users]
labels = df_norm['cluster'].values
print(f"对齐后共同用户数: {len(common_users)}")

# 归一化曲线矩阵（仅特征列）
X_norm = df_norm[[c for c in df_norm.columns if c.startswith('slot_')]].values

# 2. 形状特征计算（基于归一化曲线）
slots = np.arange(96)
morning_idx = np.arange(24, 40)   # 6:00-9:45
evening_idx = np.arange(72, 88)   # 18:00-21:45
day_idx = np.arange(32, 80)       # 8:00-19:45
night_idx = np.concatenate([np.arange(0, 32), np.arange(80, 96)])
desc_list = []
cluster_centers_norm = []
for c in range(K):
    mask = labels == c
    users_c = df_norm.index[mask]
    norm_c = df_norm.loc[users_c, [f'slot_{i}' for i in range(96)]]
    center_norm = norm_c.mean(axis=0).values
    cluster_centers_norm.append(center_norm)
    peak_slot = int(np.argmax(center_norm))
    valley_slot = int(np.argmin(center_norm))
    peak_valley_gap = center_norm.max() - center_norm.min()
    morning_mean = center_norm[morning_idx].mean()
    evening_mean = center_norm[evening_idx].mean()
    day_mean = center_norm[day_idx].mean()
    night_mean = center_norm[night_idx].mean()
    day_night_ratio = day_mean / (night_mean + 1e-8)
    desc_list.append({
        "类别": c,
        "用户数": len(users_c),
        "占比": f"{len(users_c)/len(df_norm):.1%}",
        "峰值时段": f"{peak_slot // 4:02d}:{(peak_slot % 4) * 15:02d}",
        "谷值时段": f"{valley_slot // 4:02d}:{(valley_slot % 4) * 15:02d}",
        "峰谷差(归一化)": round(peak_valley_gap, 3),
        "晨间负荷(6-10点)": round(morning_mean, 3),
        "晚间负荷(18-22点)": round(evening_mean, 3),
        "日夜比(白天/夜间)": round(day_night_ratio, 2)
    })
shape_df = pd.DataFrame(desc_list).set_index("类别")
print("\n===== 各类形状描述 =====")
print(shape_df)

# 3. 规模特征计算（基于原始负荷数据）
def compute_user_stats(series):
    # 日能耗 (kWh) = 每15分钟功率(kW)求和 * 0.25
    daily_energy = series.resample('D').apply(lambda x: x.sum() * 0.25)
    daily_mean_power = daily_energy / 24   # 等效日均功率 (kW)
    max_load = series.max()
    min_load = series.min()
    std_load = series.std()
    winter = series[series.index.month.isin([12,1,2])].mean()
    summer = series[series.index.month.isin([6,7,8])].mean()
    ws_ratio = winter / (summer + 1e-8)
    return {
        '日均能耗_kWh': daily_energy.mean(),
        '日均功率_kW': daily_mean_power.mean(),
        '最大负荷_kW': max_load,
        '最小负荷_kW': min_load,
        '负荷波动标准差_kW': std_load,
        '冬季平均功率_kW': winter,
        '夏季平均功率_kW': summer,
        '冬夏用电比': ws_ratio
    }
stats_list = []
for c in range(K):
    users = df_norm.index[labels == c]
    user_stats = []
    for user in users:
        if user in load_data.columns:
            s = load_data[user].dropna()
            if len(s) > 0:
                user_stats.append(compute_user_stats(s))
    if user_stats:
        agg = pd.DataFrame(user_stats).mean().to_dict()
    else:
        agg = {k: np.nan for k in ['日均能耗_kWh','日均功率_kW','最大负荷_kW','最小负荷_kW','负荷波动标准差_kW','冬季平均功率_kW','夏季平均功率_kW','冬夏用电比']}
    agg['类别'] = c
    stats_list.append(agg)
stats_df = pd.DataFrame(stats_list).set_index('类别').round(2)
print("\n===== 基于原始负荷的补充统计 =====")
print(stats_df)

# 4.量化用户类别
final_desc = pd.concat([shape_df, stats_df], axis=1)
energies = final_desc['日均能耗_kWh']
volatilities = final_desc['负荷波动标准差_kW']
gaps = final_desc['峰谷差(归一化)']
peak_hours = final_desc['峰值时段']
day_night = final_desc['日夜比(白天/夜间)']
ws_ratio = final_desc['冬夏用电比']
e_q1 = energies.quantile(0.33)
e_q2 = energies.quantile(0.67)
def energy_level(e):
    if e <= e_q1: return '低耗能'
    elif e <= e_q2: return '中耗能'
    else: return '高耗能'
# 波动三分位
v_q1 = volatilities.quantile(0.33)
v_q2 = volatilities.quantile(0.67)
def vol_level(v):
    if v <= v_q1: return '低波动'
    elif v <= v_q2: return '中波动'
    else: return '高波动'

def gap_desc(g):
    if g > 0.6: return '尖峰'
    elif g > 0.3: return '起伏'
    else: return '平稳'

def peak_type(peak_time):
    hour = int(peak_time.split(':')[0])
    if 5 <= hour <= 9: return '晨峰'
    elif 10 <= hour <= 14: return '午峰'
    elif 15 <= hour <= 20: return '傍晚'
    else: return '夜峰'

def dn_desc(dn):
    if dn > 1.5: return '极昼'
    elif dn > 1.2: return '偏昼'
    elif dn < 0.8: return '偏夜'
    else: return '均衡'

def season_desc(ratio):
    if ratio > 1.1: return '冬高'
    elif ratio < 0.9: return '夏高'
    else: return '均衡'

cluster_names = []
for c in final_desc.index:
    parts = [
        energy_level(energies.loc[c]),
        vol_level(volatilities.loc[c]),
        gap_desc(gaps.loc[c]),
        peak_type(peak_hours.loc[c]),
        dn_desc(day_night.loc[c]),
        season_desc(ws_ratio.loc[c])
    ]
    name = f"{parts[0]}{parts[2]}{parts[5]}型"
    cluster_names.append(name)

final_desc['类别命名'] = cluster_names

# 保存完整结果
final_desc.to_csv("cluster_full_description_ae.csv", encoding='utf-8-sig')
print("\n完整描述表已保存至 cluster_full_description_ae.csv")
print("\n===== 类别命名建议 =====")
for c in final_desc.index:
    print(f"类别 {c}: {final_desc.loc[c, '类别命名']} (用户数 {final_desc.loc[c, '用户数']}, 占比 {final_desc.loc[c, '占比']})")
print("\n✅ 完成。输出文件：")
