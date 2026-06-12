import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1、加载原始数据
print("正在加载原始数据...")
# 原始数据文件路径（请根据实际路径修改）
file_path = "LD2011_2014.txt"
# 读取数据：第一列为时间戳，其余370列为客户负荷值
df = pd.read_csv(file_path, sep=';', decimal=',', parse_dates=[0], index_col=0)
print(f"原始数据形状: {df.shape}")
print(f"时间范围: {df.index.min()} 至 {df.index.max()}")
#2、处理客户起始时间不一致
print("\n处理客户起始时间不一致...")
df = df.loc[df.index >= '2012-01-01']  # 统一起始时间
print(f"移除2011年数据后形状: {df.shape}")
# 检查是否有客户全为0（理论上可能没有实际数据），若有则剔除
all_zero_cols = df.columns[(df == 0).all()]
if len(all_zero_cols) > 0:
    print(f"剔除全零客户: {all_zero_cols.tolist()}")
    df = df.drop(columns=all_zero_cols)
#3、时间戳连续性校验与重采样
print("\n校验时间连续性...")
# 检查时间间隔是否为15分钟
time_diff = df.index.to_series().diff().dt.total_seconds().dropna()
expected_interval = 15 * 60  # 900秒
if not np.allclose(time_diff, expected_interval, atol=1):
    print("警告: 时间间隔不均为15分钟，进行重采样...")
    df = df.resample('15T').interpolate(method='linear', limit_direction='both')
else:
    print("时间连续性校验通过，所有间隔均为15分钟。")
df.to_csv("LD2011_2014_cleaned.csv", index=True)