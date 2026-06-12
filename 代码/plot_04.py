import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_path = "LD2011_2014_cleaned.csv"
output_dir = "figures"
save_fig = True
target_date = None
transform_mode = "normalize"
#y轴裁剪分位数（防止极值撑大坐标轴）
violin_ylim_quantile = 0.99
#1、数据加载与预处理
if not os.path.exists(input_path):
    raise FileNotFoundError(f"找不到输入文件: {input_path}")
df = pd.read_csv(input_path, index_col=0, parse_dates=True)
if df.empty:
    raise ValueError("输入数据为空。")
os.makedirs(output_dir, exist_ok=True)
def transform_data(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "normalize":
        col_max = data.max(axis=0).replace(0, 1)
        return data.div(col_max, axis=1)
    if mode == "log":
        return np.log1p(data)
    raise ValueError("transform_mode 必须是 'normalize' 或 'log'。")
transformed_df = transform_data(df, transform_mode)

#2、提取目标日期并转换为长格式 (Long-form)
all_dates = transformed_df.index.date
unique_dates = pd.unique(all_dates)
if target_date is None:
    target_date = str(unique_dates[0])
# 筛选当天数据
one_day = transformed_df[transformed_df.index.date == pd.to_datetime(target_date).date()]
if one_day.empty:
    raise ValueError(f"日期 {target_date} 没有对应数据。")
# 将宽表转换为 Seaborn 喜欢的长表格式
plot_data = []
for hour in range(24):
    hour_data = one_day[one_day.index.hour == hour].values.flatten()
    hour_data = hour_data[np.isfinite(hour_data)] # 剔除 NaN/Inf
    for val in hour_data:
        plot_data.append({"Hour": hour, "Value": val})
df_plot = pd.DataFrame(plot_data)

#3、可视化绘图
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.figure(figsize=(10, 5), dpi=300)
# 绘制小提琴图
ax = sns.violinplot(
    data=df_plot,
    x="Hour",
    y="Value",
    hue="Hour",
    palette="magma",
    inner="quartile",
    linewidth=1.0,
    legend=False
)
#动态计算 Y 轴范围（防止被离群值带偏）
if not df_plot.empty:
    y_limit = df_plot["Value"].quantile(violin_ylim_quantile)
    if y_limit > 0:
        plt.ylim(0, y_limit * 1.1)
plt.title(f"Hourly Distribution on {target_date} ({transform_mode.capitalize()})", 
          fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Hour of Day", fontsize=10)
ylabel = "Normalized Consumption" if transform_mode == "normalize" else "log1p(Power)"
plt.ylabel(ylabel, fontsize=10)
# 3.每3小时显示一个标签
plt.xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
sns.despine(left=True)
plt.tight_layout()
#4、保存结果
if save_fig:
    out_name = f"violin_improved_{target_date}_{transform_mode}.png"
    save_path = os.path.join(output_dir, out_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
plt.show()