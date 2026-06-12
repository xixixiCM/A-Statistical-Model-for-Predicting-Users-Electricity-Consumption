from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

try:
    from scipy.stats import energy_distance, wasserstein_distance
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False
#1、路径与参数配置
INPUT_FILE = r"prediction_results\residuals_cluster_merged.csv"
OUTPUT_DIR = r"outputs_clean_cluster"
TIMESTAMP_COL = "timestamp"
CLUSTER_COL = "cluster_id"
Y_TRUE_COL = "y_true"
Y_PRED_COL = "y_pred"
N_REFERENCE_DAYS_PER_MONTH = 7
MIN_POINTS_PER_DAY = 48
# 分布距离：energy/wasserstein
DISTANCE_METRIC = "energy"
# 阈值系数：tau=mean+lambda*std
THRESHOLD_LAMBDA = 2.0
PLOT_ENABLED = True

#2、基础工具函数
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
def load_cluster_residual_file(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    required = {TIMESTAMP_COL, CLUSTER_COL, Y_TRUE_COL, Y_PRED_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[CLUSTER_COL] = pd.to_numeric(df[CLUSTER_COL], errors="coerce")
    df[Y_TRUE_COL] = pd.to_numeric(df[Y_TRUE_COL], errors="coerce")
    df[Y_PRED_COL] = pd.to_numeric(df[Y_PRED_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL, CLUSTER_COL, Y_TRUE_COL, Y_PRED_COL])
    df[CLUSTER_COL] = df[CLUSTER_COL].astype(int)
    # 统一重新计算 residual，避免原文件 residual 列不一致
    df["residual"] = df[Y_TRUE_COL] - df[Y_PRED_COL]
    df["abs_residual"] = df["residual"].abs()
    df["relative_abs_residual"] = df["abs_residual"] / (df[Y_PRED_COL].abs() + 1e-9)
    df["date"] = df[TIMESTAMP_COL].dt.date.astype(str)
    df["month"] = df[TIMESTAMP_COL].dt.to_period("M").astype(str)
    df = df.sort_values([CLUSTER_COL, TIMESTAMP_COL]).reset_index(drop=True)
    dup_count = df.duplicated(subset=[TIMESTAMP_COL, CLUSTER_COL]).sum()
    if dup_count > 0:
        print(f"[WARNING] duplicated timestamp + cluster_id rows: {dup_count}")
    print("\n========== Loaded Data ==========")
    print("File:", path)
    print("Shape:", df.shape)
    print("Time range:", df[TIMESTAMP_COL].min(), "->", df[TIMESTAMP_COL].max())
    print("Clusters:", sorted(df[CLUSTER_COL].unique()))
    print("Rows per cluster:")
    print(df.groupby(CLUSTER_COL).size())
    return df
# 3. 生成 reference_days.csv
def build_reference_days(df: pd.DataFrame) -> pd.DataFrame:
    unique_days = (
        df[["month", "date"]]
        .drop_duplicates()
        .sort_values(["month", "date"])
        .reset_index(drop=True)
    )
    rows = []
    for month, g in unique_days.groupby("month"):
        days = list(g["date"])
        ref_days = set(days[:N_REFERENCE_DAYS_PER_MONTH])
        for d in days:
            is_reference = int(d in ref_days)
            is_test = int(d not in ref_days)
            rows.append({
                "date": d,
                "month": month,
                "is_reference": is_reference,
                "is_test": is_test,
            })
    ref_df = pd.DataFrame(rows)
    print("\n========== Reference Days ==========")
    print(ref_df.groupby("month")[["is_reference", "is_test"]].sum())
    return ref_df
# 4、构造每日残差窗口
def build_daily_windows(df: pd.DataFrame):
    windows = {}
    rows = []
    for (cid, date), g in df.groupby([CLUSTER_COL, "date"]):
        residuals = g["residual"].dropna().to_numpy(dtype=float)
        if len(residuals) < MIN_POINTS_PER_DAY:
            continue
        windows.setdefault(cid, {})[date] = residuals
        s = pd.Series(residuals)
        rows.append({
            "cluster_id": cid,
            "date": date,
            "n_points": len(residuals),
            "residual_mean": float(s.mean()),
            "residual_std": float(s.std(ddof=1)),
            "residual_min": float(s.min()),
            "residual_max": float(s.max()),
            "residual_q25": float(s.quantile(0.25)),
            "residual_q50": float(s.quantile(0.50)),
            "residual_q75": float(s.quantile(0.75)),
            "residual_skew": float(s.skew()),
            "residual_kurtosis": float(s.kurt()),
        })
    summary_df = pd.DataFrame(rows)
    print("\n========== Daily Windows ==========")
    print("Valid windows:", len(summary_df))
    print(summary_df.groupby("cluster_id")["date"].count())
    return windows, summary_df
def split_reference_and_test_windows(windows: dict, ref_df: pd.DataFrame):
    ref_days = set(ref_df.loc[ref_df["is_reference"] == 1, "date"].astype(str))
    test_days = set(ref_df.loc[ref_df["is_test"] == 1, "date"].astype(str))
    reference_windows = {}
    test_windows = {}
    for cid, day_map in windows.items():
        for date, residuals in day_map.items():
            if date in ref_days:
                reference_windows.setdefault(cid, {})[date] = residuals
            elif date in test_days:
                test_windows.setdefault(cid, {})[date] = residuals
    print("\n========== Split Windows ==========")
    for cid in sorted(windows.keys()):
        n_ref = len(reference_windows.get(cid, {}))
        n_test = len(test_windows.get(cid, {}))
        print(f"cluster {cid}: reference={n_ref}, test={n_test}")
    return reference_windows, test_windows
#5、分布距离
def energy_distance_fallback(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d_xy = np.mean(np.abs(x[:, None] - y[None, :]))
    d_xx = np.mean(np.abs(x[:, None] - x[None, :]))
    d_yy = np.mean(np.abs(y[:, None] - y[None, :]))
    val = 2 * d_xy - d_xx - d_yy
    return float(np.sqrt(max(val, 0.0)))
def distribution_distance(x: np.ndarray, y: np.ndarray, metric: str = "energy") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    if metric == "energy":
        if HAS_SCIPY:
            return float(energy_distance(x, y))
        return energy_distance_fallback(x, y)
    if metric == "wasserstein":
        if not HAS_SCIPY:
            raise RuntimeError("scipy is required for wasserstein distance.")
        return float(wasserstein_distance(x, y))
    raise ValueError(f"Unsupported distance metric: {metric}")
#6、计算类内阈值
def compute_cluster_thresholds(reference_windows: dict) -> pd.DataFrame:
    rows = []
    for cid, day_map in sorted(reference_windows.items()):
        items = list(day_map.items())
        if len(items) < 2:
            print(f"[WARNING] cluster {cid} has fewer than 2 reference days, skipped.")
            continue
        distances = []
        for (date_a, res_a), (date_b, res_b) in combinations(items, 2):
            dist = distribution_distance(res_a, res_b, DISTANCE_METRIC)
            if not np.isnan(dist):
                distances.append(dist)
        if len(distances) < 2:
            print(f"[WARNING] cluster {cid} has insufficient pairwise distances, skipped.")
            continue
        distances = np.asarray(distances, dtype=float)
        mu = float(distances.mean())
        sigma = float(distances.std(ddof=1))
        threshold = mu + THRESHOLD_LAMBDA * sigma
        rows.append({
            "cluster_id": cid,
            "distance_metric": DISTANCE_METRIC,
            "n_reference_days": len(items),
            "n_pairwise_distances": len(distances),
            "pairwise_distance_mean": mu,
            "pairwise_distance_std": sigma,
            "threshold_lambda": THRESHOLD_LAMBDA,
            "threshold": threshold,
            "pairwise_distance_min": float(distances.min()),
            "pairwise_distance_max": float(distances.max()),
        })
    thresholds_df = pd.DataFrame(rows)
    print("\n========== Cluster Thresholds ==========")
    print(thresholds_df)
    return thresholds_df
def compute_global_threshold(reference_windows: dict) -> dict:
    all_items = []
    for cid, day_map in reference_windows.items():
        for date, residuals in day_map.items():
            all_items.append((cid, date, residuals))
    if len(all_items) < 2:
        return {}
    distances = []
    for (_, _, res_a), (_, _, res_b) in combinations(all_items, 2):
        dist = distribution_distance(res_a, res_b, DISTANCE_METRIC)
        if not np.isnan(dist):
            distances.append(dist)
    distances = np.asarray(distances, dtype=float)
    mu = float(distances.mean())
    sigma = float(distances.std(ddof=1))
    threshold = mu + THRESHOLD_LAMBDA * sigma
    return {
        "distance_metric": DISTANCE_METRIC,
        "n_reference_windows": len(all_items),
        "n_pairwise_distances": len(distances),
        "pairwise_distance_mean": mu,
        "pairwise_distance_std": sigma,
        "threshold_lambda": THRESHOLD_LAMBDA,
        "threshold": threshold,
    }
#6、clean 数据异常检测
def assign_severity(score: float) -> str:
    if score <= 1.0:
        return "normal"
    if score <= 1.25:
        return "mild"
    if score <= 1.5:
        return "moderate"
    return "severe"
def detect_cluster_adaptive(test_windows: dict, reference_windows: dict, thresholds_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if thresholds_df.empty:
        return pd.DataFrame()
    threshold_map = thresholds_df.set_index("cluster_id")["threshold"].to_dict()
    for cid, day_map in sorted(test_windows.items()):
        if cid not in reference_windows:
            continue
        if cid not in threshold_map:
            continue
        pooled_reference = np.concatenate(list(reference_windows[cid].values()))
        threshold = threshold_map[cid]
        for date, residuals in sorted(day_map.items()):
            dist = distribution_distance(residuals, pooled_reference, DISTANCE_METRIC)
            score = dist / (threshold + 1e-9)
            rows.append({
                "method": "cluster_adaptive_threshold",
                "cluster_id": cid,
                "date": date,
                "distance_to_reference": dist,
                "threshold": threshold,
                "anomaly_score": score,
                "excess_distance": dist - threshold,
                "is_anomaly": int(score > 1.0),
                "severity": assign_severity(score),
            })
    return pd.DataFrame(rows)
def detect_global_baseline(test_windows: dict, reference_windows: dict) -> pd.DataFrame:
    rows = []
    global_info = compute_global_threshold(reference_windows)
    if not global_info:
        return pd.DataFrame()
    pooled_reference = []
    for day_map in reference_windows.values():
        pooled_reference.extend(list(day_map.values()))
    pooled_reference = np.concatenate(pooled_reference)
    threshold = global_info["threshold"]
    for cid, day_map in sorted(test_windows.items()):
        for date, residuals in sorted(day_map.items()):
            dist = distribution_distance(residuals, pooled_reference, DISTANCE_METRIC)
            score = dist / (threshold + 1e-9)
            rows.append({
                "method": "global_distribution_threshold",
                "cluster_id": cid,
                "date": date,
                "distance_to_reference": dist,
                "threshold": threshold,
                "anomaly_score": score,
                "excess_distance": dist - threshold,
                "is_anomaly": int(score > 1.0),
                "severity": assign_severity(score),
            })
    return pd.DataFrame(rows)
def summarize_clean_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, cid), g in scores_df.groupby(["method", "cluster_id"]):
        n = len(g)
        n_anom = int(g["is_anomaly"].sum())
        rows.append({
            "method": method,
            "cluster_id": cid,
            "n_test_days": n,
            "n_anomalies": n_anom,
            "anomaly_rate": n_anom / (n + 1e-9),
            "mean_score": float(g["anomaly_score"].mean()),
            "max_score": float(g["anomaly_score"].max()),
            "mean_distance": float(g["distance_to_reference"].mean()),
            "mean_threshold": float(g["threshold"].mean()),
        })
    summary_df = pd.DataFrame(rows)
    print("\n========== Clean Baseline Summary ==========")
    print(summary_df)
    return summary_df
# 8、可视化
def plot_cluster_thresholds(thresholds_df: pd.DataFrame, output_dir: str):
    if not PLOT_ENABLED or not HAS_PLT or thresholds_df.empty:
        return
    fig_dir = Path(output_dir) / "figures"
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4))
    plt.bar(thresholds_df["cluster_id"].astype(str), thresholds_df["threshold"])
    plt.xlabel("cluster_id")
    plt.ylabel("threshold")
    plt.title("Cluster adaptive thresholds")
    plt.tight_layout()
    plt.savefig(fig_dir / "cluster_thresholds_bar.png", dpi=150)
    plt.close()
def plot_anomaly_timeline(scores_df: pd.DataFrame, output_dir: str):
    if not PLOT_ENABLED or not HAS_PLT or scores_df.empty:
        return
    fig_dir = Path(output_dir) / "figures"
    ensure_dir(fig_dir)
    for (method, cid), g in scores_df.groupby(["method", "cluster_id"]):
        g = g.sort_values("date")
        plt.figure(figsize=(12, 4))
        plt.plot(g["date"], g["anomaly_score"], marker="o", linewidth=1)
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("date")
        plt.ylabel("anomaly_score")
        plt.title(f"{method} | cluster {cid}")
        plt.tight_layout()
        filename = f"anomaly_score_timeline_{method}_cluster_{cid}.png"
        plt.savefig(fig_dir / filename, dpi=150)
        plt.close()
def plot_anomaly_heatmap(scores_df: pd.DataFrame, output_dir: str):
    if not PLOT_ENABLED or not HAS_PLT or scores_df.empty:
        return
    fig_dir = Path(output_dir) / "figures"
    ensure_dir(fig_dir)
    main_df = scores_df[scores_df["method"] == "cluster_adaptive_threshold"].copy()
    if main_df.empty:
        return
    pivot = main_df.pivot_table(
        index="cluster_id",
        columns="date",
        values="anomaly_score",
        aggfunc="mean"
    )
    plt.figure(figsize=(14, 4))
    plt.imshow(pivot.fillna(0).values, aspect="auto")
    plt.colorbar(label="anomaly_score")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.xlabel("date")
    plt.ylabel("cluster_id")
    plt.title("Cluster adaptive anomaly score heatmap")
    plt.tight_layout()
    plt.savefig(fig_dir / "anomaly_heatmap.png", dpi=150)
    plt.close()
# 9、主流程
def main():
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "figures")
    df = load_cluster_residual_file(INPUT_FILE)
    ref_df = build_reference_days(df)
    ref_df.to_csv(output_dir / "reference_days.csv", index=False, encoding="utf-8-sig")
    #①保存点级残差
    df.to_csv(output_dir / "01_point_residuals.csv", index=False, encoding="utf-8-sig")
    #构造日级残差窗口
    windows, window_summary_df = build_daily_windows(df)
    window_summary_df.to_csv(
        output_dir / "02_window_residual_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    #②划分 reference/test 窗口
    reference_windows, test_windows = split_reference_and_test_windows(windows, ref_df)
    #计算类内阈值
    thresholds_df = compute_cluster_thresholds(reference_windows)
    thresholds_df.to_csv(
        output_dir / "03_cluster_thresholds.csv",
        index=False,
        encoding="utf-8-sig"
    )
    #③类内阈值 clean 检测
    cluster_scores_df = detect_cluster_adaptive(
        test_windows=test_windows,
        reference_windows=reference_windows,
        thresholds_df=thresholds_df
    )
    #④全局阈值 clean 检测
    global_scores_df = detect_global_baseline(
        test_windows=test_windows,
        reference_windows=reference_windows
    )
    scores_df = pd.concat(
        [cluster_scores_df, global_scores_df],
        ignore_index=True
    )
    scores_df.to_csv(
        output_dir / "04_clean_anomaly_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )
    #⑤汇总 clean 异常比例
    summary_df = summarize_clean_scores(scores_df)
    summary_df.to_csv(
        output_dir / "05_clean_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    #⑥可视化
    plot_cluster_thresholds(thresholds_df, output_dir)
    plot_anomaly_timeline(scores_df, output_dir)
    plot_anomaly_heatmap(scores_df, output_dir)
    print("\n========== Done ==========")
    print("Output dir:", output_dir)
    print("Main files:")
    print(" - reference_days.csv")
    print(" - 03_cluster_thresholds.csv")
    print(" - 04_clean_anomaly_scores.csv")
    print(" - 05_clean_summary.csv")
if __name__ == "__main__":
    main()