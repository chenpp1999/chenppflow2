import numpy as np
import pandas as pd
import torch


def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    print(feat_path)
    print(feat_path)
    print(feat_path)
    print(feat_path)

    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    print(adj_path)
    return adj



def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val

    # train_size = int(time_len * split_ratio)
    # train_data = data[:train_size]
    # test_data = data[train_size:time_len]

    train_size_all = int(time_len * split_ratio)
    # train_size_all = int(time_len * 0.7)
    train_data_all = data[:train_size_all]
    test_data_all = data[train_size_all:time_len]

    # train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    # for i in range(len(train_data) - seq_len - pre_len):
    #     train_X.append(np.array(train_data[i : i + seq_len]))
    #     train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    # for i in range(len(test_data) - seq_len - pre_len):
    #     test_X.append(np.array(test_data[i : i + seq_len]))
    #     test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))

    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    train_X_t, test_X_t = list(), list()
    train_X_d, test_X_d = list(), list()

    for i in range(len(train_data_all) - seq_len - pre_len):
        train_X.append(np.array(train_data_all[i: i + seq_len]))
        train_X_d.append(np.array(train_data_all[i+pre_len: i + seq_len+pre_len]))
        train_Y.append(np.array(train_data_all[i + seq_len: i + seq_len + pre_len]))
    # 去除第一天,最后一天
    train_X_t = train_X[288:]
    train_X_d = train_X_d[:-288]
    train_Y = train_Y[288:]

    # print(train_X_d[888])
    train_X_td, test_X_td = list(), list()
    for i in range(len(train_X_d)):
        train_X_td.append(np.row_stack((train_X_t[i], train_X_d[i])))

    for i in range(len(test_data_all) - seq_len - pre_len):
        test_X.append(np.array(test_data_all[i: i + seq_len]))
        test_X_d.append(np.array(test_data_all[i + pre_len: i + seq_len + pre_len]))
        test_Y.append(np.array(test_data_all[i + seq_len: i + seq_len + pre_len]))
    # 去除第一天,最后一天
    test_X_t = test_X
    test_X_d = train_X_d[-288:] + test_X_d[:-288]
    test_Y = test_Y
    for i in range(len(test_X_d)):
        test_X_td.append(np.row_stack((test_X_t[i], test_X_d[i])))

    # print(len(train_data_all))
    # print(len(train_X_t))
    # print(len(train_X_d))
    # print(len(train_Y))
    # print(len(test_X_t))
    # print(len(test_X_d))
    # print(len(test_Y))
    # print("拼接：", len(train_X_td))
    # print("拼接：", len(test_X_td))

    return np.array(train_X_td), np.array(train_Y), np.array(test_X_td), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
