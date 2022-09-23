import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


import torch

a = torch.eye(207, 1)
print(a)

feat = load_features('data/sz_speed.csv')
feat_weather = load_features('data/sz_weather.csv')
# feat = load_features('data/sz_speed.csv')

print("矩阵形状：", feat.shape)
print("时间线长度：", len(feat[:, 0]))
print('城市节点数：', len(feat[1, :]))

print("天气矩阵形状：", feat_weather.shape)
print("天气时间线长度：", len(feat_weather[:, 0]))
print('天气城市节点数：', len(feat_weather[2, :]))


y = [i for i in feat[:, 1]]
# print(type(x))
x = [i for i in range(len(y))]
print(len(x))
plt.plot(x, y, 'r-', color='#4169E1', alpha=1, linewidth=1, label='流量数')
plt.legend(loc="upper right")
plt.xlabel('时间轴')
plt.ylabel('流量轴')
plt.show()