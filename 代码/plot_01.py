import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1、Input requirements
raw_path = "daily_profile_raw.csv"
norm_path = "daily_profile_norm.csv"

#2、Load data and select first 4 users
raw_df = pd.read_csv(raw_path, index_col=0, encoding="utf-8-sig")
norm_df = pd.read_csv(norm_path, index_col=0, encoding="utf-8-sig")
user_ids = raw_df.index[:4].tolist()
raw_profiles = raw_df.loc[user_ids].values
norm_profiles = norm_df.loc[user_ids].values
slots = raw_profiles.shape[1]

#3、Plot 1x2 layout
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=300, sharex=True)
colors = ["#77295D", "#C34FA2", "#77B7F0", "#5364C0"]
for i, user_id in enumerate(user_ids):
    axes[0].plot(range(slots), raw_profiles[i], color=colors[i], linewidth=1.5, label=f"User {user_id}")
    axes[1].plot(range(slots), norm_profiles[i], color=colors[i], linewidth=1.5, label=f"User {user_id}")
x_ticks = np.linspace(0, 96, 7)
x_labels = ["0:00", "4:00", "8:00", "12:00", "16:00", "20:00", "24:00"]
for ax in axes:
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
axes[0].set_title("Raw Daily Profiles", fontsize=9)
axes[1].set_title("Normalized Daily Profiles", fontsize=9)
axes[0].set_ylabel("Power Consumption (kW)", fontsize=9)
axes[1].set_ylabel("Normalized Consumption", fontsize=9)
axes[1].legend(
    loc="upper right",
    fontsize=7,
    frameon=True,
    framealpha=1.0,
    facecolor="white",
    edgecolor="none"
)
fig.supxlabel("Time of Day (15-min Slots)", fontsize=9)
plt.tight_layout()
plt.savefig("figure2.png", dpi=300)
plt.show()
