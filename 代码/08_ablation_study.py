import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100
#消融后的特征构建
def build_features_ablation(series, idx, window=96, ablation=None):
    N = len(series)
    features = pd.DataFrame(index=idx)
    # ==================== 1. 时域特征 ====================
    if ablation != 'no_temporal':
        features['lag_1'] = np.roll(series, 1)
        features['lag_4'] = np.roll(series, 4)
        features['lag_96'] = np.roll(series, 96)
        features['lag_672'] = np.roll(series, 672)
        rolling = pd.Series(series).rolling(window=window, min_periods=1)
        features['rolling_mean_96'] = rolling.mean().values
        features['rolling_std_96'] = rolling.std().values
    # ==================== 2. 派生特征 ====================
    if ablation != 'no_derived':
        delta = np.diff(series, prepend=series[0])
        features['jump_ratio'] = np.abs(delta) / (np.abs(series) + 1e-8)
        threshold = np.percentile(series, 95)
        high_count = pd.Series(series).rolling(window, min_periods=1).apply(
            lambda x: np.mean(x > threshold))
        features['high_ratio'] = high_count.values
    # ==================== 3. 周期性编码特征 ====================
    if ablation != 'no_cyclic':
        hours = idx.hour
        weekdays = idx.weekday
        months = idx.month
        features['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        features['weekday_sin'] = np.sin(2 * np.pi * weekdays / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * weekdays / 7)
        features['month_sin'] = np.sin(2 * np.pi * months / 12)
        features['month_cos'] = np.cos(2 * np.pi * months / 12)
    # 统一截断
    features = features.iloc[672:]
    y = series[672:]
    features['load'] = y
    return features
def train_rf(X, y, test_ratio=0.2):
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return y_test.values, y_pred
def run_ablation(ablation_name, ablation_key):
    """运行一组消融实验并返回指标"""
    cluster_test_actuals = []
    cluster_test_preds = []
    min_test_len = None
    for cid, series in enumerate(cluster_loads):
        feat = build_features_ablation(series.values, series.index, ablation=ablation_key)
        Xc = feat.drop('load', axis=1)
        yc = feat['load']
        y_true_c, y_pred_c = train_rf(Xc, yc)
        cluster_test_actuals.append(y_true_c)
        cluster_test_preds.append(y_pred_c)
        if min_test_len is None or len(y_true_c) < min_test_len:
            min_test_len = len(y_true_c)
    cluster_test_actuals = [a[:min_test_len] for a in cluster_test_actuals]
    cluster_test_preds = [p[:min_test_len] for p in cluster_test_preds]
    actual = np.sum(cluster_test_actuals, axis=0)
    pred = np.sum(cluster_test_preds, axis=0)
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    sp = smape(actual, pred)
    return actual, pred, (mae, rmse, r2, sp)
# ==================== 主程序 ====================
if __name__ == "__main__":
    # 加载4个簇负荷（不变）
    cluster_loads = []
    for cid in range(4):
        path = f"cluster_load_data/cluster_load_{cid}.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        cluster_loads.append(df.sum(axis=1))
    # 逐步移除每一类特征
    ablation_groups = [
        ("全特征（基准）", None),
        ("无时序特征", "no_temporal"),
        ("无派生特征", "no_derived"),
        ("无周期特征", "no_cyclic"),
    ]
    results = []
    actual_all = None
    pred_dict = {}
    # ==================== 运行并保存====================
    print("========== 分层预测消融实验 ==========\n")
    for name, key in ablation_groups:
        actual, pred, metrics = run_ablation(name, key)
        mae, rmse, r2, sp = metrics
        print(f"{name:15} | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f} | SMAPE={sp:.2f}%")
        results.append([name, mae, rmse, r2, sp])
        if actual_all is None:
            actual_all = actual
        pred_dict[name] = pred
    df_res = pd.DataFrame(results, columns=['特征组合', 'MAE', 'RMSE', 'R2', 'SMAPE(%)'])
    df_res.to_csv("ablation_study_hierarchical.csv", index=False, encoding='utf-8-sig')
    print("\n消融实验结果已保存至 ablation_study_hierarchical.csv")
    names_short = ['全特征', '无时序', '无派生', '无周期']
    maes = df_res['MAE'].values
    rmses = df_res['RMSE'].values
    r2s = df_res['R2'].values
    smapes = df_res['SMAPE(%)'].values
    # ====================可视化图绘制====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    show_len = 200
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = [g[0] for g in ablation_groups]
    for i, (name, _) in enumerate(ablation_groups):
        axes[i].plot(actual_all[:show_len], color='black', linewidth=1.2, alpha=0.9, label='真实值')
        axes[i].plot(pred_dict[name][:show_len], color=colors[i], linewidth=1.4, linestyle='--', label=labels[i])
        axes[i].set_title(f'{labels[i]}', fontsize=13, fontweight='bold')
        axes[i].legend(loc='upper right', fontsize=11)
        axes[i].grid(False)
        axes[i].set_ylabel('负荷 (kW)', fontsize=11)
    axes[2].set_xlabel('时间点 (15min)', fontsize=11)
    axes[3].set_xlabel('时间点 (15min)', fontsize=11)
    plt.suptitle('分层预测消融实验对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ablation_4subplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n消融实验全部完成！")