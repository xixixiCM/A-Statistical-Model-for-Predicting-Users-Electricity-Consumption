import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100
def train_and_predict_cluster(feature_file, cluster_id, test_ratio=0.2):
    df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
    X = df.drop('load', axis=1)
    y = df['load']
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    timestamps = X_test.index
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    sp = smape(y_test, y_pred)
    result = pd.DataFrame({
        'timestamp': timestamps,
        'cluster_id': cluster_id,
        'y_true': y_test.values,
        'y_pred': y_pred,
        'residual': y_test.values - y_pred
    })
    return result, {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'SMAPE': sp}
#可视化图绘制
def plot_cluster_prediction(result_df, cid, save_path):
    plt.figure(figsize=(14, 5))
    plt.plot(result_df['timestamp'], result_df['y_true'], label='真实负荷', linewidth=1.5, color='#1f77b4')
    plt.plot(result_df['timestamp'], result_df['y_pred'], label='预测负荷', linewidth=1.5, color='#ff7f0e',
             linestyle='--')
    plt.title(f'簇 {cid} 预测值 vs 真实值', fontweight='bold', fontsize=14)
    plt.xlabel('时间')
    plt.ylabel('负荷 (kW)')
    plt.legend()
    if cid == 3:
        plt.ylim(20000, 50000)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'cluster_{cid}_prediction.png'), bbox_inches='tight')
    plt.close()
def plot_residual_sequence(result_df, cid, save_path):
    plt.figure(figsize=(14, 5))
    plt.plot(result_df['timestamp'], result_df['residual'], color='#2ca02c', linewidth=1, label='残差')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f'簇 {cid} 残差时序分布', fontweight='bold', fontsize=14)
    plt.xlabel('时间')
    plt.ylabel('残差')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'cluster_{cid}_residual.png'), bbox_inches='tight')
    plt.close()
def plot_metrics_bar(metrics_df, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    metrics = ['MAE', 'RMSE', 'R2', 'SMAPE']
    titles = ['MAE 对比', 'RMSE 对比', '$R^2$ 对比', 'SMAPE(%) 对比']
    for i, (m, t) in enumerate(zip(metrics, titles)):
        axes[i].bar(metrics_df['cluster_id'], metrics_df[m], color=colors, alpha=0.8)
        axes[i].set_title(t, fontweight='bold')
        axes[i].set_xlabel('簇编号')
        axes[i].set_ylabel(m)
    plt.suptitle('各簇预测模型性能指标对比', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'cluster_metrics_comparison.png'), bbox_inches='tight')
    plt.close()
def plot_residual_histogram(merged_df, save_path):
    plt.figure(figsize=(10, 5))
    residuals = merged_df['residual']
    plt.hist(residuals, bins=50, color='#9467bd', alpha=0.7, edgecolor='black', linewidth=0.3)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title('合并残差分布直方图', fontweight='bold', fontsize=14)
    plt.xlabel('残差')
    plt.ylabel('频次')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'merged_residual_histogram.png'), bbox_inches='tight')
    plt.close()

def main():
    input_dir = "cluster_features"
    output_dir = "prediction_results"
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    all_metrics = []
    for cid in range(4):
        feature_file = os.path.join(input_dir, f"features_cluster_{cid}.csv")
        if not os.path.exists(feature_file):
            print(f"警告: {feature_file} 不存在，跳过簇 {cid}")
            continue
        print(f"处理簇 {cid} ...")
        result_df, metrics = train_and_predict_cluster(feature_file, cid)
        out_file = os.path.join(output_dir, f"residuals_cluster_{cid}.csv")
        result_df.to_csv(out_file, index=False, encoding='utf-8-sig')
        plot_cluster_prediction(result_df, cid, output_dir)
        plot_residual_sequence(result_df, cid, output_dir)
        all_results.append(result_df)
        all_metrics.append({'cluster_id': cid, **metrics})
    if all_results:
        merged = pd.concat(all_results, ignore_index=True)
        merged.to_csv(os.path.join(output_dir, "residuals_cluster_merged.csv"), index=False, encoding='utf-8-sig')
        plot_residual_histogram(merged, output_dir)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, "cluster_forecast_metrics.csv"), index=False, encoding='utf-8-sig')
    plot_metrics_bar(metrics_df, output_dir)
    print("\n全部完成！")
if __name__ == "__main__":
    main()