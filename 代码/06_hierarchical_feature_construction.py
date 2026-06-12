import pandas as pd
import numpy as np
import os


def build_cluster_features(series, idx_timestamp):
    N = len(series)
    window = 96
    features = pd.DataFrame(index=idx_timestamp)

    # ----- 1. 构建时域特征 -----
    features['lag_1'] = np.roll(series, 1)
    features['lag_4'] = np.roll(series, 4)
    features['lag_96'] = np.roll(series, 96)
    features['lag_672'] = np.roll(series, 672)

    rolling = pd.Series(series).rolling(window=window, min_periods=1)
    features['rolling_mean_96'] = rolling.mean().values
    features['rolling_std_96'] = rolling.std().values

    # ----- 2. 构建负荷波动特征 -----
    delta = np.diff(series, prepend=series[0])
    features['jump_ratio'] = np.abs(delta) / (np.abs(series) + 1e-8)
    threshold = np.percentile(series, 95)
    high_count = pd.Series(series).rolling(window, min_periods=1).apply(lambda x: np.mean(x > threshold))
    features['high_ratio'] = high_count.values

    # ----- 3. 构建周期性编码特征（基于时间戳）-----
    # 提取时间分量
    hours = idx_timestamp.hour
    weekdays = idx_timestamp.weekday
    months = idx_timestamp.month
    features['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    features['weekday_sin'] = np.sin(2 * np.pi * weekdays / 7)
    features['weekday_cos'] = np.cos(2 * np.pi * weekdays / 7)
    features['month_sin'] = np.sin(2 * np.pi * months / 12)
    features['month_cos'] = np.cos(2 * np.pi * months / 12)
    # 丢弃前672个点（滞后边界和窗口边界）
    features = features.iloc[672:]
    y = series[672:]
    features['load'] = y
    return features

def main():
    input_dir = "cluster_load_data"
    output_dir = "cluster_features"
    os.makedirs(output_dir, exist_ok=True)
    for cid in range(4):
        file_path = os.path.join(input_dir, f"cluster_load_{cid}.csv")
        if not os.path.exists(file_path):
            print(f"警告：{file_path} 不存在，跳过簇 {cid}")
            continue
        # 读取簇负荷（时间索引作为列，或者自动解析）
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # 计算该簇的总负荷（如果文件中已经是单列则不需要 sum，但簇文件是每个用户一列，需要求和）
        load_series = df.sum(axis=1).values
        idx = df.index
        print(f"处理簇 {cid}: 序列长度 {len(load_series)}")
        features = build_cluster_features(load_series, idx)
        out_file = os.path.join(output_dir, f"features_cluster_{cid}.csv")
        features.to_csv(out_file, encoding='utf-8-sig')
        print(f"  已保存至 {out_file}, 形状 {features.shape}")
if __name__ == "__main__":
    main()