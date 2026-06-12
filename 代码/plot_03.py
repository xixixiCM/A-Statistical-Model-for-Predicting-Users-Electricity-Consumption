import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Input requirements
input_path = "LD2011_2014_cleaned.csv"
output_dir = "figures"
save_fig = True
target_date = None
violin_ylim_quantile = 0.99

# 2. Load data
df = pd.read_csv(input_path, index_col=0, parse_dates=True)
if df.empty:
    raise ValueError("Input data is empty.")

os.makedirs(output_dir, exist_ok=True)
# 3. Heatmap: total load across full period
series_total = df.sum(axis=1)
heat_df = pd.DataFrame({"load": series_total.values}, index=series_total.index)
heat_df["date"] = heat_df.index.date
heat_df["slot"] = heat_df.index.hour * 4 + (heat_df.index.minute // 15)
heat_pivot = heat_df.pivot_table(index="date", columns="slot", values="load", aggfunc="mean")
heat_pivot = heat_pivot.reindex(columns=range(96))
heat_pivot = heat_pivot.dropna()
plt.figure(figsize=(7.5, 4.5), dpi=300)
plt.imshow(heat_pivot.values, aspect="auto", cmap="viridis")
plt.colorbar(label="Total Load (kW)")
x_ticks = np.linspace(0, 96, 7)
x_labels = ["0:00", "4:00", "8:00", "12:00", "16:00", "20:00", "24:00"]
plt.xticks(x_ticks, x_labels, fontsize=8)
# Sparse y-ticks to avoid clutter
if len(heat_pivot.index) >= 5:
    y_ticks = np.linspace(0, len(heat_pivot.index) - 1, 5).astype(int)
    y_labels = [str(heat_pivot.index[i]) for i in y_ticks]
    plt.yticks(y_ticks, y_labels, fontsize=7)
else:
    plt.yticks(range(len(heat_pivot.index)), heat_pivot.index, fontsize=7)
plt.title("Full-Period Load Heatmap (Total Load)", fontsize=10)
plt.xlabel("Time of Day (15-min Slots)", fontsize=9)
plt.ylabel("Date", fontsize=9)
plt.tight_layout()
if save_fig:
    plt.savefig(os.path.join(output_dir, "heatmap_total_load.png"), dpi=300)
plt.show()

# 4. Violin plot: 24-hour distribution for one day
all_dates = heat_pivot.index.tolist()
if target_date is None:
    target_date = str(all_dates[0])
# Filter one day from the original data
one_day = df[df.index.date == pd.to_datetime(target_date).date()]
if one_day.empty:
    raise ValueError(f"No data found for target_date={target_date}.")
# Collect distributions for each hour (0-23)
violin_data = []
for hour in range(24):
    hour_df = one_day[(one_day.index.hour == hour)]
    if hour_df.empty:
        violin_data.append(np.array([]))
        continue
    # Use all users and all 15-min slots within the hour
    violin_data.append(hour_df.values.flatten())
plt.figure(figsize=(7.5, 4.0), dpi=300)
parts = plt.violinplot(
    violin_data,
    positions=np.arange(24),
    widths=0.8,
    showmeans=False,
    showmedians=True,
    showextrema=False
)
for body in parts["bodies"]:
    body.set_facecolor("#C34FA2")
    body.set_edgecolor("none")
    body.set_alpha(0.75)
if "cmedians" in parts:
    parts["cmedians"].set_color("#77295D")
    parts["cmedians"].set_linewidth(1.2)
all_vals = np.concatenate([v for v in violin_data if v.size > 0])
all_vals = all_vals[np.isfinite(all_vals)]
if all_vals.size > 0:
    y_max = float(np.quantile(all_vals, violin_ylim_quantile))
    if y_max > 0:
        plt.ylim(0, y_max)

plt.xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)], fontsize=8)
plt.yticks(fontsize=8)
plt.title(f"Hourly Distribution on {target_date}", fontsize=10)
plt.xlabel("Hour of Day", fontsize=9)
plt.ylabel("Power Consumption (kW)", fontsize=9)
plt.tight_layout()
if save_fig:
    plt.savefig(os.path.join(output_dir, "violin_one_day_hourly.png"), dpi=300)
plt.show()
