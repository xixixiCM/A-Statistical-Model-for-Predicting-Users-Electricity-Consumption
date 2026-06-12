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

# 1. 参数设置
ATTACK_DIR = r"attacks_cluster"
CLEAN_SCORE_PATH = r"outputs_clean_cluster/04_clean_anomaly_scores.csv"
OUTPUT_DIR = r"attack_detection_results"
TIMESTAMP_COL = "timestamp"
CLUSTER_COL = "cluster_id"
Y_TRUE_COL = "y_true"
Y_PRED_COL = "y_pred"
MIN_POINTS_PER_DAY = 48
DISTANCE_METRIC = "energy"
THRESHOLD_LAMBDA = 2.0
PLOT_ENABLED = True

#2、工具函数
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
def extract_attack_name(pred_path: Path) -> str:
    stem = pred_path.stem
    prefix = "predictions_attack_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem
def load_attack_predictions(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Attack prediction file not found: {path}")
    df = pd.read_csv(path)
    required = {TIMESTAMP_COL, CLUSTER_COL, Y_TRUE_COL, Y_PRED_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[CLUSTER_COL] = pd.to_numeric(df[CLUSTER_COL], errors="coerce")
    df[Y_TRUE_COL] = pd.to_numeric(df[Y_TRUE_COL], errors="coerce")
    df[Y_PRED_COL] = pd.to_numeric(df[Y_PRED_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL, CLUSTER_COL, Y_TRUE_COL, Y_PRED_COL])
    df[CLUSTER_COL] = df[CLUSTER_COL].astype(int)
    df["date"] = df[TIMESTAMP_COL].dt.date.astype(str)
    df["month"] = df[TIMESTAMP_COL].dt.to_period("M").astype(str)
    # 重新计算攻击后的残差，确保一致
    df["residual"] = df[Y_TRUE_COL] - df[Y_PRED_COL]
    df["abs_residual"] = df["residual"].abs()
    df["relative_abs_residual"] = df["abs_residual"] / (df[Y_PRED_COL].abs() + 1e-9)
    df = df.sort_values([CLUSTER_COL, TIMESTAMP_COL]).reset_index(drop=True)
    return df
def load_reference_days(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"reference_days.csv not found: {path}")
    ref_df = pd.read_csv(path)
    required = {"date", "is_reference", "is_test"}
    missing = required - set(ref_df.columns)
    if missing:
        raise ValueError(f"reference_days.csv missing columns: {missing}")
    ref_df["date"] = ref_df["date"].astype(str)
    ref_df["is_reference"] = ref_df["is_reference"].astype(int)
    ref_df["is_test"] = ref_df["is_test"].astype(int)
    return ref_df
def load_labels(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    label_df = pd.read_csv(path)
    required = {"date", CLUSTER_COL, "label"}
    missing = required - set(label_df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    label_df["date"] = label_df["date"].astype(str)
    label_df[CLUSTER_COL] = pd.to_numeric(label_df[CLUSTER_COL], errors="coerce")
    label_df["label"] = pd.to_numeric(label_df["label"], errors="coerce")
    label_df = label_df.dropna(subset=["date", CLUSTER_COL, "label"])
    label_df[CLUSTER_COL] = label_df[CLUSTER_COL].astype(int)
    label_df["label"] = label_df["label"].astype(int)
    # 只评估测试日；reference days 是正常基准，不参与攻击检测评价
    if "is_test" in label_df.columns:
        label_df = label_df[label_df["is_test"] == 1].copy()
    return label_df
# 3、日窗口构造
def build_daily_windows(df):
    windows = {}
    for (cid, date), g in df.groupby([CLUSTER_COL, "date"]):
        residuals = g["residual"].dropna().to_numpy(dtype=float)
        if len(residuals) < MIN_POINTS_PER_DAY:
            continue
        windows.setdefault(cid, {})[date] = residuals
    return windows
def split_reference_and_test_windows(windows, ref_df):
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
    return reference_windows, test_windows
# 4、分布距离
def energy_distance_fallback(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d_xy = np.mean(np.abs(x[:, None] - y[None, :]))
    d_xx = np.mean(np.abs(x[:, None] - x[None, :]))
    d_yy = np.mean(np.abs(y[:, None] - y[None, :]))
    val = 2 * d_xy - d_xx - d_yy
    return float(np.sqrt(max(val, 0.0)))
def distribution_distance(x, y, metric="energy"):
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
    raise ValueError(f"Unsupported metric: {metric}")

#5、阈值计算
def compute_cluster_thresholds(reference_windows):
    rows = []
    for cid, day_map in sorted(reference_windows.items()):
        items = list(day_map.items())
        if len(items) < 2:
            continue
        distances = []
        for (_, res_a), (_, res_b) in combinations(items, 2):
            dist = distribution_distance(res_a, res_b, DISTANCE_METRIC)
            if not np.isnan(dist):
                distances.append(dist)
        if len(distances) < 2:
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
        })
    return pd.DataFrame(rows)
def compute_global_threshold(reference_windows):
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
#6、异常检测
def assign_severity(score):
    if score <= 1.0:
        return "normal"
    if score <= 1.25:
        return "mild"
    if score <= 1.5:
        return "moderate"
    return "severe"
def detect_cluster_adaptive(test_windows, reference_windows, thresholds_df, attack_name):
    rows = []
    if thresholds_df.empty:
        return pd.DataFrame()
    threshold_map = thresholds_df.set_index("cluster_id")["threshold"].to_dict()
    for cid, day_map in sorted(test_windows.items()):
        if cid not in reference_windows:
            continue
        if cid not in threshold_map:
            continue
        pooled_ref = np.concatenate(list(reference_windows[cid].values()))
        threshold = threshold_map[cid]
        for date, residuals in sorted(day_map.items()):
            dist = distribution_distance(residuals, pooled_ref, DISTANCE_METRIC)
            score = dist / (threshold + 1e-9)
            rows.append({
                "attack_name": attack_name,
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
def detect_global_baseline(test_windows, reference_windows, attack_name):
    rows = []
    global_info = compute_global_threshold(reference_windows)
    if not global_info:
        return pd.DataFrame()
    pooled_ref = []
    for day_map in reference_windows.values():
        pooled_ref.extend(list(day_map.values()))
    pooled_ref = np.concatenate(pooled_ref)
    threshold = global_info["threshold"]
    for cid, day_map in sorted(test_windows.items()):
        for date, residuals in sorted(day_map.items()):
            dist = distribution_distance(residuals, pooled_ref, DISTANCE_METRIC)
            score = dist / (threshold + 1e-9)
            rows.append({
                "attack_name": attack_name,
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
#7、指标计算
def safe_div(a, b):
    if b == 0:
        return np.nan
    return a / b
def load_clean_fpr(clean_score_path):
    path = Path(clean_score_path)
    if not path.exists():
        print(f"[WARNING] Clean score file not found: {path}")
        print("[WARNING] clean_fpr and paper_balanced_accuracy will be NaN.")
        return pd.DataFrame()
    clean_scores = pd.read_csv(path)
    required = {"method", "cluster_id", "is_anomaly"}
    missing = required - set(clean_scores.columns)
    if missing:
        print(f"[WARNING] Clean score file missing columns: {missing}")
        return pd.DataFrame()
    clean_scores["cluster_id"] = clean_scores["cluster_id"].astype(int)
    clean_scores["is_anomaly"] = clean_scores["is_anomaly"].astype(int)
    rows = []
    for (method, cid), g in clean_scores.groupby(["method", "cluster_id"]):
        rows.append({
            "method": method,
            "cluster_id": str(cid),
            "clean_fpr": float(g["is_anomaly"].mean()),
            "clean_n_days": len(g),
        })
    for method, g in clean_scores.groupby("method"):
        rows.append({
            "method": method,
            "cluster_id": "ALL",
            "clean_fpr": float(g["is_anomaly"].mean()),
            "clean_n_days": len(g),
        })
    return pd.DataFrame(rows)
def evaluate_scores(scores_df, labels_df, attack_name, clean_fpr_df):
    labels = labels_df[["date", CLUSTER_COL, "label"]].copy()
    labels["date"] = labels["date"].astype(str)
    labels[CLUSTER_COL] = labels[CLUSTER_COL].astype(int)
    merged = scores_df.merge(
        labels,
        on=["date", CLUSTER_COL],
        how="left"
    )
    # 没匹配到的默认正常
    merged["label"] = merged["label"].fillna(0).astype(int)
    rows = []
    for (method, cid), g in merged.groupby(["method", CLUSTER_COL]):
        rows.append(compute_metric_row(
            attack_name=attack_name,
            method=method,
            cluster_id=str(cid),
            g=g,
            clean_fpr_df=clean_fpr_df
        ))
    for method, g in merged.groupby("method"):
        rows.append(compute_metric_row(
            attack_name=attack_name,
            method=method,
            cluster_id="ALL",
            g=g,
            clean_fpr_df=clean_fpr_df
        ))
    return pd.DataFrame(rows)
def compute_metric_row(attack_name, method, cluster_id, g, clean_fpr_df):
    y_true = g["label"].astype(int)
    y_pred = g["is_anomaly"].astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if not np.isnan(precision) and not np.isnan(recall) else np.nan
    # 当前攻击文件内部的 FPR；如果攻击测试日几乎全是 label=1，这个值会 NaN
    internal_fpr = safe_div(fp, fp + tn)
    detection_rate = recall
    clean_fpr = np.nan
    if not clean_fpr_df.empty:
        match = clean_fpr_df[
            (clean_fpr_df["method"] == method)
            & (clean_fpr_df["cluster_id"] == cluster_id)
        ]
        if not match.empty:
            clean_fpr = float(match["clean_fpr"].iloc[0])
    if np.isnan(clean_fpr) or np.isnan(detection_rate):
        paper_balanced_accuracy = np.nan
    else:
        paper_balanced_accuracy = 0.5 * (detection_rate + (1.0 - clean_fpr))
    return {
        "attack_name": attack_name,
        "method": method,
        "cluster_id": cluster_id,
        "n_eval_days": len(g),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall_detection_rate": recall,
        "f1": f1,
        "internal_fpr": internal_fpr,
        "clean_fpr": clean_fpr,
        "paper_balanced_accuracy": paper_balanced_accuracy,
        "mean_anomaly_score": float(g["anomaly_score"].mean()),
        "max_anomaly_score": float(g["anomaly_score"].max()),
        "n_predicted_anomalies": int(g["is_anomaly"].sum()),
        "n_true_anomalies": int(g["label"].sum()),
    }
# 8. 汇总与可视化
def summarize_average_metrics(all_metrics_df):
    rows = []
    all_rows = all_metrics_df[all_metrics_df["cluster_id"] == "ALL"].copy()
    for method, g in all_rows.groupby("method"):
        rows.append({
            "method": method,
            "n_attacks": g["attack_name"].nunique(),
            "avg_detection_rate": float(g["recall_detection_rate"].mean()),
            "avg_clean_fpr": float(g["clean_fpr"].mean()),
            "avg_paper_balanced_accuracy": float(g["paper_balanced_accuracy"].mean()),
            "avg_precision": float(g["precision"].mean()),
            "avg_f1": float(g["f1"].mean()),
            "avg_mean_anomaly_score": float(g["mean_anomaly_score"].mean()),
        })
    return pd.DataFrame(rows)
def plot_detection_rate_by_attack(all_metrics_df, output_dir):
    if not PLOT_ENABLED or not HAS_PLT:
        return
    fig_dir = Path(output_dir) / "figures"
    ensure_dir(fig_dir)
    df = all_metrics_df[all_metrics_df["cluster_id"] == "ALL"].copy()
    if df.empty:
        return
    pivot = df.pivot_table(
        index="attack_name",
        columns="method",
        values="recall_detection_rate",
        aggfunc="mean"
    )
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Detection Rate / Recall")
    plt.ylim(0, 1.05)
    plt.title("Detection rate by attack type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "detection_rate_by_attack.png", dpi=150)
    plt.close()
def plot_balanced_accuracy_by_attack(all_metrics_df, output_dir):
    if not PLOT_ENABLED or not HAS_PLT:
        return
    fig_dir = Path(output_dir) / "figures"
    ensure_dir(fig_dir)
    df = all_metrics_df[all_metrics_df["cluster_id"] == "ALL"].copy()
    if df.empty:
        return
    pivot = df.pivot_table(
        index="attack_name",
        columns="method",
        values="paper_balanced_accuracy",
        aggfunc="mean"
    )
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Paper-style Balanced Accuracy")
    plt.ylim(0, 1.05)
    plt.title("Balanced accuracy by attack type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "balanced_accuracy_by_attack.png", dpi=150)
    plt.close()

# 9、主流程
def main():
    attack_dir = Path(ATTACK_DIR)
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "figures")
    ref_path = attack_dir / "reference_days.csv"
    ref_df = load_reference_days(ref_path)
    clean_fpr_df = load_clean_fpr(CLEAN_SCORE_PATH)
    if not clean_fpr_df.empty:
        clean_fpr_df.to_csv(output_dir / "clean_fpr_used.csv", index=False, encoding="utf-8-sig")
    pred_files = sorted(attack_dir.glob("predictions_attack_*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions_attack_*.csv found in {attack_dir}")
    all_scores = []
    all_metrics = []
    print("\n========== Attack Detection Start ==========")
    print("Attack dir:", attack_dir)
    print("Output dir:", output_dir)
    print("Attack files:", len(pred_files))
    for pred_path in pred_files:
        attack_name = extract_attack_name(pred_path)
        label_path = attack_dir / f"labels_attack_{attack_name}.csv"
        print(f"\n========== Processing Attack: {attack_name} ==========")
        print("Prediction:", pred_path.name)
        print("Labels:    ", label_path.name)
        df = load_attack_predictions(pred_path)
        labels_df = load_labels(label_path)
        windows = build_daily_windows(df)
        reference_windows, test_windows = split_reference_and_test_windows(windows, ref_df)
        thresholds_df = compute_cluster_thresholds(reference_windows)
        thresholds_df.to_csv(
            output_dir / f"thresholds_attack_{attack_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )
        cluster_scores = detect_cluster_adaptive(
            test_windows=test_windows,
            reference_windows=reference_windows,
            thresholds_df=thresholds_df,
            attack_name=attack_name
        )
        global_scores = detect_global_baseline(
            test_windows=test_windows,
            reference_windows=reference_windows,
            attack_name=attack_name
        )
        scores_df = pd.concat([cluster_scores, global_scores], ignore_index=True)
        scores_df.to_csv(
            output_dir / f"scores_attack_{attack_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )
        metrics_df = evaluate_scores(
            scores_df=scores_df,
            labels_df=labels_df,
            attack_name=attack_name,
            clean_fpr_df=clean_fpr_df
        )
        metrics_df.to_csv(
            output_dir / f"metrics_attack_{attack_name}.csv",
            index=False,
            encoding="utf-8-sig"
        )
        all_scores.append(scores_df)
        all_metrics.append(metrics_df)
        print(metrics_df[metrics_df["cluster_id"] == "ALL"][
            [
                "attack_name",
                "method",
                "recall_detection_rate",
                "clean_fpr",
                "paper_balanced_accuracy",
                "n_predicted_anomalies",
                "n_true_anomalies",
            ]
        ])
    all_scores_df = pd.concat(all_scores, ignore_index=True)
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_scores_df.to_csv(
        output_dir / "all_attack_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )
    all_metrics_df.to_csv(
        output_dir / "all_attack_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )
    avg_metrics_df = summarize_average_metrics(all_metrics_df)
    avg_metrics_df.to_csv(
        output_dir / "average_metrics_by_method.csv",
        index=False,
        encoding="utf-8-sig"
    )
    plot_detection_rate_by_attack(all_metrics_df, output_dir)
    plot_balanced_accuracy_by_attack(all_metrics_df, output_dir)
    print("\n========== Average Metrics ==========")
    print(avg_metrics_df)
    print("\n========== Done ==========")
    print("Output dir:", output_dir)
    print("Main outputs:")
    print(" - all_attack_metrics.csv")
    print(" - average_metrics_by_method.csv")
    print(" - all_attack_scores.csv")
if __name__ == "__main__":
    main()