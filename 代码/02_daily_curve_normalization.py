import pandas as pd
import numpy as np

# 1、读取数据
df = pd.read_csv('LD2011_2014_cleaned.csv', index_col=0, parse_dates=True)
print("原始数据形状:", df.shape)
print("时间范围:", df.index.min(), "->", df.index.max())

#2、构造时间槽位（0-95，每15分钟一个点）
# 提取小时和分钟
hours = df.index.hour
minutes = df.index.minute
slot = hours * 4 + minutes // 15   # 结果范围 0~95
# 将槽位添加到DataFrame中
df_temp = df.copy()
df_temp['slot'] = slot

#3、按槽位分组，计算每个用户每个槽位的平均值（所有天的平均）
daily_profile = df_temp.groupby('slot').mean()
#将daily_profile 形状为(96, n_users)转置为 (n_users, 96)
daily_profile = daily_profile.T
daily_profile.index.name = 'user_id'
daily_profile.columns = [f'slot_{i}' for i in range(96)]
print("每日平均曲线形状:", daily_profile.shape)
print("前5行示例:\n", daily_profile.head())
daily_profile.to_csv("daily_profile_raw.csv", encoding='utf-8-sig')

# 4、行归一化（每个用户除以其自身最大值）
def row_max_normalize(df):
    """对DataFrame每行除以该行的最大值"""
    row_max = df.max(axis=1)
    # 避免除以0（如果整行全为0，则保持不变）
    row_max[row_max == 0] = 1
    return df.div(row_max, axis=0)
daily_profile_norm = row_max_normalize(daily_profile)
daily_profile_norm.to_csv("daily_profile_norm.csv", encoding='utf-8-sig')
print("归一化后曲线范围: 每行最小={:.4f}, 最大={:.4f}".format(
    daily_profile_norm.min().min(), daily_profile_norm.max().max()))
print("✅ 已完成，输出文件：daily_profile_raw.csv 和 daily_profile_norm.csv")