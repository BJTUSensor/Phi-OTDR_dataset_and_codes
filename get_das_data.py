import scipy.io as scio
import numpy as np
from feature_extraction import feature_extraction

def get_diff_data(data):                       # 数据差分
    m = data.shape[0]        # 10000
    n = data.shape[1]        # 12
    data_diff = np.empty([m-1, n]).astype(int)   # 9999,12
    for i in range(m-1):
        data_diff[i, :] = data[i+1, :] - data[i, :]
    return data_diff

def get_feature_list(data):                     # 10000,12
    sample_feature_list = np.zeros([12, 16])    # 提特征 12,16
    for i in range(12):
        f_data = data[:, i]                         # 10000,1 / 9999,1
        feature_list = feature_extraction(f_data)   # 16
        sample_feature_list[i, :] = feature_list
    return sample_feature_list

def get_das_data(rootpath, labelpath):
    datapath = rootpath
    file = open(labelpath)
    name_list = []
    for f in file:
        name_list.append(f)
    temp = np.empty([len(name_list), 12, 32])
    label_temp = np.empty(len(name_list))
    for i in range(len(name_list)):
        path = datapath + name_list[i].split(' ')[0]
        rawdata = scio.loadmat(path)['data']              # 原始数据 10000,12
        diffdata = get_diff_data(rawdata)               # 差分数据 9999,12
        rawdata_sample_feature_list = get_feature_list(rawdata)   # 原始数据特征 12,16
        diffdata_sample_feature_list = get_feature_list(diffdata)  # 差分数据特征 12,16
        sample_feature_list = np.concatenate((rawdata_sample_feature_list, diffdata_sample_feature_list), axis=1)   # 两类特征合并 12,32
        feature_data = np.reshape(sample_feature_list, (1, sample_feature_list.shape[0], sample_feature_list.shape[1]))
        temp[i, :, :] = feature_data
        label = int(name_list[i].split(' ')[1])
        label_temp[i] = label
    temp = temp.reshape(len(name_list), -1)  # 样本量，12*32（展平）
    return temp, label_temp

