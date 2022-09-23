from tasks import SupervisedForecastTask
import os
import torch
from models import TGCN
import utils.data
from utils.data import functions
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# def load_ckpt(
#     ckpt_path, ckpt_dict
#     ):
#     restore_ckpt = {}
#     best_val = epoch_start = 0
#     if os.path.exists(ckpt_path):
#         ckpt = torch.load(ckpt_path)
#         for (k, v) in ckpt_dict.items():
#             model_dict = v.state_dict()
#             if k in ckpt :
#                 for i,j in ckpt[k].items():
#                     if "cab1.conv3" not in i and "cab2.conv3" not in i and "cab3.conv3" not in i and "cab4.conv3" not in i and "clf_conv" not in i:
#                         restore_ckpt[i] = j
#                 model_dict.update(restore_ckpt)
#                 print(model_dict)
#                 v.load_state_dict(model_dict)
#         best_val = ckpt.get('best_val', 0)
#         epoch_start = ckpt.get('epoch_start', 0)
#         # logger.info(" Found checkpoint at {} with best_val {:.4f} at epoch {}".
#         #     format(
#         #         ckpt_path, best_val, epoch_start
#         #     ))
#     return best_val, epoch_start

if __name__ == '__main__':
    # ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_29_attention207_yuanshuju\\checkpoints\\epoch=2949-step=17700.ckpt'
    # ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\lightning_logs\\version_30_attention156_sz_mae2.705\\checkpoints\\epoch=2890-step=26019.ckpt'
    # ckpt_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\lightning_logs\\version_53_双attention_效果差_mae3.05\\checkpoints\\epoch=556-step=5013.ckpt'
    # ckpt_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\lightning_logs\\version_57_单attention156_舍弃前1000数据_mae2.78\\checkpoints\\epoch=2996-step=14985.ckpt'
    # ckpt_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\lightning_logs\\version_58_单attention156_全数据_seed3407_mae2.71\\checkpoints\\epoch=2737-step=24642.ckpt'
    # ckpt_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\lightning_logs\\version_59_单attention156_全数据_seed3407_batch32_mae2.68\\checkpoints\\epoch=719-step=51120.ckpt'
    # ckpt_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\lightning_logs\\version_65_Resnet_mae2.80\\checkpoints\\epoch=2889-step=26010.ckpt'
    # ckpt_path = '/T-GCN-PyTorch-sz/lightning_logs/version_54_舍弃前1000数据_双attention_mae2.87\\checkpoints\\epoch=2984-step=14925.ckpt'
    # ckpt_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\lightning_logs\\version_67\\checkpoints\\epoch=855-step=60776.ckpt'
    ckpt_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\lightning_logs\\version_70_bigru_mae1.53\\checkpoints\\epoch=2918-step=26271.ckpt'

    adj_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\data\\sz_adj.csv'
    feat_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\data\\sz_speed.csv'
    weather_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch-sz\\data\\sz_weather.csv'

    adj = utils.data.functions.load_adjacency_matrix(adj_path)

    myNet = SupervisedForecastTask(model=TGCN(adj=adj, hidden_dim=64))
    myNet = myNet.load_from_checkpoint(ckpt_path)
    myNet = myNet.eval()

    _weather = functions.load_weather(weather_path)
    _feat = functions.load_features(feat_path)
    max_val = np.max(_feat)

    train_dataset, val_dataset = functions.generate_torch_datasets(data=_feat, weather_data=_weather, seq_len=12, pre_len=3, split_ratio=0.8)

    # gen_val = DataLoader(val_dataset, shuffle=True, batch_size=24, num_workers=12)
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=len(val_dataset), num_workers=12) # 加了shuffle精度上升了


    y_list, pre_list, x_list = list(), list(), list()
    for iteration, batch in enumerate(gen_val):
        x, y = batch
        # print(x)
        print(x.size())
        num_nodes = x.size(2)
        predictions = myNet(x)
        print(predictions.size())
        py = predictions.mean(axis=2)
        print("py:", py.size())
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        print(predictions.size())
        print(y.size())
        yp = y.mean(axis=1)
        print('yp:',yp.size())
        y = y.reshape((-1, y.size(2)))

        feat_max_val: float = 1.0
        predictions = predictions * feat_max_val
        y = y * feat_max_val
        pre_list.append(py)
        y_list.append(yp)



    # print(pre_list[1].size())
    pre = pre_list[0].detach().numpy()
    yy = y_list[0].detach().numpy()
    # print(len(pre[:, 32]))
    p_list = pre[:, 26]*max_val   #2 3 80
    yyy_list = yy[:, 26]*max_val
    print(type(pre_list))
    print(pre_list)


    x = [i for i in range(len(p_list))]
    plt.plot(x, p_list, 'r-', color='#FF0000', alpha=1, linewidth=1, label='predict')
    plt.plot(x, yyy_list, 'r-', color='#4169E1', alpha=1, linewidth=1, label='label')
    plt.legend(loc="upper right")
    plt.xlabel('time')
    plt.ylabel('flow')
    # plt.title("qianyitian_288")  # 标题
    # plt.title("qianyitian_340")  # 标题
    # plt.title("shuangshurutong")  # 标题
    # plt.title("shuangshuruyi_xiuzheng")  # 标题
    # plt.title("shuangshuruyi_shujuxiuzheng2")  # 标题
    # plt.title("shuangshuruyi_shujuxiuzhengln")  # 标题
    # plt.title("shuangshuruyi_shujuxiuzhengln_attention207")  # 标题
    plt.title("version_29_attention207_yuanshuju")  # 标题
    # plt.title("shuangshuruyi_shujuxiuzhengln_attention120")  # 标题
    # plt.title("shuangshuruyi")  # 标题
    plt.show()

