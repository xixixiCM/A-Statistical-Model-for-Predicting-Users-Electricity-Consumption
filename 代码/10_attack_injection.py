from pathlib import Path
import numpy as np
import pandas as pd

#1、参数设置
CLEAN_PRED_PATH = r"prediction_results/residuals_cluster_merged.csv"
REFERENCE_DAYS_PATH = r"outputs_clean_cluster/reference_days.csv"
OUTPUT_DIR = r"attacks_cluster"
TIMESTAMP_COL = "timestamp"
CLUSTER_COL = "cluster_id"
Y_TRUE_COL = "y_true"
Y_PRED_COL = "y_pred"
RANDOM_SEED = 42
N_REFERENCE_DAYS_PER_MONTH = 7

# 2、攻击配置
ATTACK_CONFIGS = [
    {
        "name": "scale_down",
        "type": "scale_down",
        "alpha": 0.6,
    },
    {
        "name": "random_scale_down",
        "type": "random_scale_down",
        "alpha_low": 0.4,
        "alpha_high": 0.8,
    },
    {
        "name": "zero",
        "type": "zero",
    },
    {
        "name": "random_zero",
        "type": "random_zero",
        "zero_prob": 0.3,
    },
    {
        "name": "mean_replace",
        "type": "mean_replace",
    },
    {
        "name": "low_day_replace",
        "type": "low_day_replace",
    },
]

# 3、读取与检查数据
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
def load_clean_predictions(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean prediction file not found: {path}")
    df = pd.read_csv(path)
    required = {TIMESTAMP_COL, CLUSTER_COL, Y_TRUE_COL, Y_PRED_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Clean file missing columns: {missing}")
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df[CLUSTER_COL] = pd.to_numeric(df[CLUSTER_COL], errors="coerce")
    df[Y_TRUE_COL] = pd.to_numeric(df[Y_TRUE_COL], errors="coerce")
    df[Y_PRED_COL] = pd.to_numeric(df[Y_PRED_COL], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL, CLUSTER_COL, Y_TRUE_COL, Y_PRED_COL])
    df[CLUSTER_COL] = df[CLUSTER_COL].astype(int)
    df = df.sort_values([CLUSTER_COL, TIMESTAMP_COL]).reset_index(drop=True)
    df["date"] = df[TIMESTAMP_COL].dt.date.astype(str)
    df["month"] = df[TIMESTAMP_COL].dt.to_period("M").astype(str)
    df["time_of_day"] = df[TIMESTAMP_COL].dt.strftime("%H:%M:%S")
    # 保留干净真实值
    df["y_clean"] = df[Y_TRUE_COL].astype(float)
    print("\n========== Loaded Clean Predictions ==========")
    print("File:", path)
    print("Shape:", df.shape)
    print("Time range:", df[TIMESTAMP_COL].min(), "->", df[TIMESTAMP_COL].max())
    print("Clusters:", sorted(df[CLUSTER_COL].unique()))
    print("Rows per cluster:")
    print(df.groupby(CLUSTER_COL).size())
    return df
def build_reference_days_from_data(df: pd.DataFrame) -> pd.DataFrame:
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
            rows.append({
                "date": d,
                "month": month,
                "is_reference": int(d in ref_days),
                "is_test": int(d not in ref_days),
            })
    return pd.DataFrame(rows)
def load_or_build_reference_days(df: pd.DataFrame, path: str) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        ref_df = pd.read_csv(path)
    else:
        print(f"[WARNING] reference_days.csv not found: {path}")
        print("[INFO] Build reference_days automatically.")
        ref_df = build_reference_days_from_data(df)
    required = {"date", "is_reference", "is_test"}
    missing = required - set(ref_df.columns)
    if missing:
        raise ValueError(f"reference_days file missing columns: {missing}")
    ref_df["date"] = ref_df["date"].astype(str)
    ref_df["is_reference"] = ref_df["is_reference"].astype(int)
    ref_df["is_test"] = ref_df["is_test"].astype(int)
    print("\n========== Reference Days ==========")
    if "month" in ref_df.columns:
        print(ref_df.groupby("month")[["is_reference", "is_test"]].sum())
    else:
        print(ref_df[["is_reference", "is_test"]].sum())
    return ref_df
def attach_reference_flags(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(
        ref_df[["date", "is_reference", "is_test"]],
        on="date",
        how="left"
    )
    if out["is_reference"].isna().any() or out["is_test"].isna().any():
        missing_dates = out.loc[out["is_reference"].isna(), "date"].unique()
        raise ValueError(f"Some dates are not found in reference_days.csv: {missing_dates[:10]}")
    out["is_reference"] = out["is_reference"].astype(int)
    out["is_test"] = out["is_test"].astype(int)
    return out
#4、攻击方法
def apply_scale_down(df: pd.DataFrame, alpha: float) -> pd.Series:
    y_attack = df["y_clean"].copy()
    mask = df["is_test"] == 1
    y_attack.loc[mask] = alpha * df.loc[mask, "y_clean"]
    return y_attack
def apply_random_scale_down(
    df: pd.DataFrame,
    alpha_low: float,
    alpha_high: float,
    rng: np.random.Generator
) -> pd.Series:
    y_attack = df["y_clean"].copy()
    mask = df["is_test"] == 1
    alphas = rng.uniform(alpha_low, alpha_high, size=mask.sum())
    y_attack.loc[mask] = df.loc[mask, "y_clean"].to_numpy() * alphas
    return y_attack
def apply_zero_attack(df: pd.DataFrame) -> pd.Series:
    y_attack = df["y_clean"].copy()
    mask = df["is_test"] == 1
    y_attack.loc[mask] = 0.0
    return y_attack
def apply_random_zero(
    df: pd.DataFrame,
    zero_prob: float,
    rng: np.random.Generator
) -> tuple[pd.Series, pd.Series]:
    y_attack = df["y_clean"].copy()
    test_mask = df["is_test"] == 1
    random_mask = pd.Series(False, index=df.index)
    selected = rng.random(size=len(df)) < zero_prob
    random_mask.loc[test_mask] = selected[test_mask.to_numpy()]
    y_attack.loc[random_mask] = 0.0
    return y_attack, random_mask.astype(int)
def apply_mean_replace(df: pd.DataFrame) -> pd.Series:
    y_attack = df["y_clean"].copy()
    test_df = df[df["is_test"] == 1].copy()
    daily_mean = (
        test_df
        .groupby([CLUSTER_COL, "date"])["y_clean"]
        .transform("mean")
    )
    y_attack.loc[test_df.index] = daily_mean.to_numpy()
    return y_attack
def apply_low_day_replace(df: pd.DataFrame) -> pd.Series:
    y_attack = df["y_clean"].copy()
    for cid, g_cluster in df.groupby(CLUSTER_COL):
        ref_part = g_cluster[g_cluster["is_reference"] == 1].copy()
        test_part = g_cluster[g_cluster["is_test"] == 1].copy()
        if ref_part.empty or test_part.empty:
            continue
        daily_sum = (
            ref_part
            .groupby("date")["y_clean"]
            .sum()
            .sort_values()
        )
        low_day = daily_sum.index[0]
        low_profile = (
            ref_part[ref_part["date"] == low_day]
            [["time_of_day", "y_clean"]]
            .rename(columns={"y_clean": "low_day_value"})
        )
        low_mean = float(low_profile["low_day_value"].mean())
        tmp = test_part[["time_of_day"]].merge(
            low_profile,
            on="time_of_day",
            how="left"
        )
        replacement = tmp["low_day_value"].fillna(low_mean).to_numpy()
        y_attack.loc[test_part.index] = replacement
    return y_attack
def inject_attack(
    clean_df: pd.DataFrame,
    attack_cfg: dict,
    rng: np.random.Generator
) -> pd.DataFrame:
    df = clean_df.copy()
    attack_name = attack_cfg["name"]
    attack_type = attack_cfg["type"]
    # 默认：所有 test points 被攻击，reference points 不攻击
    is_attacked_point = (df["is_test"] == 1).astype(int)
    if attack_type == "scale_down":
        y_attack = apply_scale_down(df, alpha=attack_cfg["alpha"])
    elif attack_type == "random_scale_down":
        y_attack = apply_random_scale_down(
            df,
            alpha_low=attack_cfg["alpha_low"],
            alpha_high=attack_cfg["alpha_high"],
            rng=rng
        )
    elif attack_type == "zero":
        y_attack = apply_zero_attack(df)
    elif attack_type == "random_zero":
        y_attack, is_attacked_point = apply_random_zero(
            df,
            zero_prob=attack_cfg["zero_prob"],
            rng=rng
        )
    elif attack_type == "mean_replace":
        y_attack = apply_mean_replace(df)
    elif attack_type == "low_day_replace":
        y_attack = apply_low_day_replace(df)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    # reference days 必须保持 clean
    ref_mask = df["is_reference"] == 1
    y_attack.loc[ref_mask] = df.loc[ref_mask, "y_clean"]
    is_attacked_point.loc[ref_mask] = 0
    df[Y_TRUE_COL] = y_attack.astype(float)
    df["residual"] = df[Y_TRUE_COL] - df[Y_PRED_COL]
    df["abs_residual"] = df["residual"].abs()
    df["relative_abs_residual"] = df["abs_residual"] / (df[Y_PRED_COL].abs() + 1e-9)
    df["attack_type"] = attack_name
    df["is_attacked_point"] = is_attacked_point.astype(int)
    # 输出列排序
    cols = [
        TIMESTAMP_COL,
        CLUSTER_COL,
        "date",
        "month",
        "is_reference",
        "is_test",
        "attack_type",
        "is_attacked_point",
        "y_clean",
        Y_TRUE_COL,
        Y_PRED_COL,
        "residual",
        "abs_residual",
        "relative_abs_residual",
    ]
    df = df[cols].sort_values([CLUSTER_COL, TIMESTAMP_COL]).reset_index(drop=True)
    return df
# 5、标签生成
def build_window_labels(attacked_df: pd.DataFrame, attack_name: str) -> pd.DataFrame:
    label_df = (
        attacked_df
        .groupby(["date", CLUSTER_COL], as_index=False)
        .agg(
            label=("is_attacked_point", "max"),
            n_points=("is_attacked_point", "size"),
            n_attacked_points=("is_attacked_point", "sum"),
            is_reference=("is_reference", "max"),
            is_test=("is_test", "max"),
        )
    )
    label_df["attack_type"] = np.where(
        label_df["label"] == 1,
        attack_name,
        "normal"
    )
    label_df = label_df[
        [
            "date",
            CLUSTER_COL,
            "label",
            "attack_type",
            "n_points",
            "n_attacked_points",
            "is_reference",
            "is_test",
        ]
    ]
    return label_df
def summarize_attack(attacked_df: pd.DataFrame, label_df: pd.DataFrame, attack_name: str) -> dict:
    point_attack_rate = attacked_df["is_attacked_point"].mean()
    day_attack_rate = label_df["label"].mean()
    test_label_df = label_df[label_df["is_test"] == 1]
    test_day_attack_rate = test_label_df["label"].mean()
    return {
        "attack_name": attack_name,
        "n_rows": len(attacked_df),
        "n_cluster_days": len(label_df),
        "point_attack_rate_all": point_attack_rate,
        "cluster_day_attack_rate_all": day_attack_rate,
        "cluster_day_attack_rate_test": test_day_attack_rate,
        "n_attacked_points": int(attacked_df["is_attacked_point"].sum()),
        "n_attacked_cluster_days": int(label_df["label"].sum()),
    }
# 6、主流程
def main():
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)
    rng = np.random.default_rng(RANDOM_SEED)
    clean_df = load_clean_predictions(CLEAN_PRED_PATH)
    ref_df = load_or_build_reference_days(clean_df, REFERENCE_DAYS_PATH)
    clean_df = attach_reference_flags(clean_df, ref_df)
    # 保存reference_days，保证攻击实验目录完整
    ref_df.to_csv(output_dir / "reference_days.csv", index=False, encoding="utf-8-sig")
    manifest_rows = []
    for attack_cfg in ATTACK_CONFIGS:
        attack_name = attack_cfg["name"]
        print(f"\n========== Inject Attack: {attack_name} ==========")
        attacked_df = inject_attack(
            clean_df=clean_df,
            attack_cfg=attack_cfg,
            rng=rng
        )
        label_df = build_window_labels(attacked_df, attack_name=attack_name)
        pred_path = output_dir / f"predictions_attack_{attack_name}.csv"
        label_path = output_dir / f"labels_attack_{attack_name}.csv"
        attacked_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        label_df.to_csv(label_path, index=False, encoding="utf-8-sig")
        summary = summarize_attack(attacked_df, label_df, attack_name)
        manifest_rows.append(summary)
        print("Saved prediction:", pred_path)
        print("Saved labels:    ", label_path)
        print("Attack summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / "attack_manifest.csv", index=False, encoding="utf-8-sig")
    print("\n========== Done ==========")
    print("Output dir:", output_dir)
    print("Generated files:")
    for p in sorted(output_dir.glob("*.csv")):
        print(" -", p.name)
if __name__ == "__main__":
    main()