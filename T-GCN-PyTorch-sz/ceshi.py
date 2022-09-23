import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.data import functions
plt.rcParams['font.sans-serif'] = ['SimHei']  ##中文乱码问题！
plt.rcParams['axes.unicode_minus'] = False  # 横坐标负号显示问题！

adj_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\data\\sz_adj.csv'
feat_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\data\\sz_speed.csv'
weather_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\data\\sz_weather.csv'
_feat = functions.load_features(feat_path)
max_val = np.max(_feat)

###初始化参数
my_seed = 369  # 随便给出个随机种子

# sell_data = np.array(
#     [2800, 2811, 2832, 2850, 2880, 2910, 2960, 3023, 3039, 3056, 3138, 3150, 3198, 3100, 3029, 2950, 2989, 3012, 3050,
#      3142, 3252, 3342, 3365, 3385, 3340, 3410, 3443, 3428, 3554, 3615, 3646, 3614, 3574, 3635, 3738, 3764, 3788, 3820,
#      3840, 3875, 3900, 3942, 4000, 4021, 4055])

S_sell_data = pd.Series(_feat[0]).diff(1)  ##差分
revisedata = S_sell_data.max()
sell_datanormalization = S_sell_data / revisedata  ##数据规范化

print(S_sell_data)

sell_datanormalization =sell_datanormalization*revisedata

print(sell_datanormalization.cumsum()+2800)