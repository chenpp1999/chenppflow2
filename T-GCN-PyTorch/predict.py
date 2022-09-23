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
    # ckpt_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_17_双输入同_前一天_340_乱序版\\checkpoints\\epoch=2998-step=14995.ckpt'
    # ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_1_双输入同\\checkpoints\\epoch=2232-step=111650.ckpt'
    # ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_11_双输入异\\checkpoints\\epoch=2953-step=17724.ckpt'
    # ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_16_双输入同_前一天\\checkpoints\\epoch=1606-step=17677.ckpt'
    # ckpt_path ='C:\\Users\\15315\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_18_双输入异_修正\\checkpoints\\epoch=2725-step=16356.ckpt'
    # ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_20_双输入异_数据平方\\checkpoints\\epoch=2986-step=17922.ckpt'
    # ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_21_双输入异_数据ln\\checkpoints\\epoch=2950-step=17706.ckpt'
    # ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_23_双输入异_数据ln_attention207\\checkpoints\\epoch=2875-step=17256.ckpt'
    # ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_25_双输入异_数据ln_attention120\\checkpoints\\epoch=2998-step=17994.ckpt'
    ckpt_path ='C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\lightning_logs\\version_29_attention207_yuanshuju\\checkpoints\\epoch=2949-step=17700.ckpt'

    adj_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\data\\los_adj.csv'
    feat_path = 'C:\\Users\\15315\\Desktop\\conda cp\\T-GCN-PyTorch\\data\\los_speed.csv'
    adj = utils.data.functions.load_adjacency_matrix(adj_path)

    myNet = SupervisedForecastTask(model=TGCN(adj=adj, hidden_dim=64))
    myNet = myNet.load_from_checkpoint(ckpt_path)
    myNet = myNet.eval()

    _feat = functions.load_features(feat_path)
    max_val = np.max(_feat)

    train_dataset, val_dataset = functions.generate_torch_datasets(data=_feat, seq_len=12, pre_len=3, split_ratio=0.8)

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
    p_list = pre[:, 0]*max_val
    yyy_list = yy[:, 0]*max_val
    print(type(pre_list))
    print(pre_list)


    x = [i for i in range(len(p_list))]
    plt.plot(x, p_list, 'r-', color='#FF0000', alpha=1, linewidth=2, label='predict')
    plt.plot(x, yyy_list, 'r-', color='#4169E1', alpha=1, linewidth=2, label='label')
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

