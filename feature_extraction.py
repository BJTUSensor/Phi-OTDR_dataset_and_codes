import numpy as np
import pandas as pd

# 特征提取函数，输入数据：差分（9999,1），原始数据（10000,1）。返回（16,1）。
def feature_extraction(data):  # data (9999,1)
    def fft_fft(data):
        fft_trans = np.abs(np.fft.fft(data))
        freq_spectrum = fft_trans[1:int(np.floor(len(data) * 1.0 / 2)) + 1]
        _freq_sum_ = np.sum(freq_spectrum)
        return freq_spectrum, _freq_sum_
    freq_spectrum, _freq_sum_ = fft_fft(data)
    # 最大值
    dif_max = max(data)
    # 最小值
    dif_min = min(data)
    # 峰峰值
    dif_pk = int(dif_max)-int(dif_min)
    # 均值
    dif_mean = data.mean()
    # 方差
    dif_var = data.var()
    # 标准差
    dif_std = data.std()
    # 能量
    dif_energy = np.sum(freq_spectrum ** 2) / len(freq_spectrum)
    # 均方根
    dif_rms = np.sqrt(pow(dif_mean, 2) + pow(dif_std, 2))
    # 整流平均值
    dif_arv = abs(data).mean()
    # 波形因子
    dif_boxing = dif_rms / (abs(data).mean())
    # 脉冲因子   if
    dif_maichong = (max(data)) / (abs(data).mean())
    # 峰值因子   cf
    dif_fengzhi = (max(data)) / dif_rms
    # 裕度因子   cl
    sum = 0
    for i in range(len(data)):
        sum += np.sqrt(abs(data[i]))
    dif_yudu = max(data) / pow(sum / (len(data)), 2)
    # 峰度因子
    dif_kurt = pd.Series(data).kurt()
    # 峭度因子
    dif_qiaodu = (np.sum([x ** 4 for x in data]) / len(data)) / pow(dif_rms, 4)
    # 信息熵
    pr_freq = freq_spectrum * 1.0 / _freq_sum_
    dif_entropy = -1 * np.sum([np.log2(p + 1e-5) * p for p in pr_freq])

    feature_list = [round(dif_max, 3), round(dif_min, 3), round(dif_pk, 3), round(dif_mean, 3),
                    round(dif_energy, 3), round(dif_var, 3), round(dif_std, 3), round(dif_rms, 3),
                    round(dif_arv, 3), round(dif_boxing, 3), round(dif_maichong, 3), round(dif_fengzhi, 3),
                    round(dif_yudu, 3), round(dif_kurt, 3), round(dif_qiaodu, 3), round(dif_entropy, 3)
                    ]
    feature_list = np.array(feature_list)
    return feature_list

