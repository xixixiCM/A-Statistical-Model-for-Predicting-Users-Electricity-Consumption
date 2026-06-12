import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#1、Load data and select first 4 users
input_path = "LD2011_2014_cleaned.csv"
sample_days = 30
random_seed = 42
df = pd.read_csv(input_path, index_col=0, parse_dates=True)
user_ids = df.columns[:4].tolist()

#2、Helper: build daily matrix (days x 96 slots)
def build_daily_matrix(series):
    tmp = pd.DataFrame({"value": series.values}, index=series.index)
    tmp["date"] = tmp.index.date
    tmp["slot"] = tmp.index.hour * 4 + (tmp.index.minute // 15)
    pivot = tmp.pivot_table(index="date", columns="slot", values="value", aggfunc="mean")
    pivot = pivot.reindex(columns=range(96))
    pivot = pivot.dropna()
    return pivot
#3、Plot 2x2 subplots with gray daily lines
fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300, sharex=True, sharey=False)
axes = axes.flatten()
colors = ["#77295D", "#C34FA2", "#77B7F0", "#5364C0"]
legend_handles = []
legend_labels = []
for i, ax in enumerate(axes):
    user_id = user_ids[i]
    series = df[user_id].dropna()
    daily_df = build_daily_matrix(series)
    if len(daily_df) > sample_days:
        daily_df = daily_df.sample(n=sample_days, random_state=random_seed)
    #Gray background lines (random days)
    for day_curve in daily_df.values:
        ax.plot(range(96), day_curve, color="#CFCDDA", alpha=0.4, linewidth=1)
    #Bright average line (mean of gray lines)
    mean_curve = daily_df.mean(axis=0).values
    line, = ax.plot(range(96), mean_curve, color=colors[i], linewidth=1, label=f"User {user_id}")
    legend_handles.append(line)
    legend_labels.append(f"User {user_id}")
    ax.grid(False)
x_ticks = np.linspace(0, 96, 7)
x_labels = ["0:00", "4:00", "8:00", "12:00", "16:00", "20:00", "24:00"]
for ax in axes:
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=6)
    ax.tick_params(axis="y", labelsize=5)
legend = fig.legend(
    legend_handles,
    legend_labels,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    fontsize=6,
    frameon=True,
    framealpha=0.9,
    facecolor="white",
    edgecolor="none"
)
legend.set_zorder(10)
fig.supxlabel("Time of Day (15-min Slots)", fontsize=6)
fig.supylabel("Power Consumption (kW)", fontsize=6, x=0.02)
plt.tight_layout()
plt.savefig("figure1.png", dpi=300)
plt.show()
