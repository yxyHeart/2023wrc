import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
import math
import scipy
from scipy import signal
from numpy import (sin, sqrt, einsum)
import scipy.linalg as LA
from scipy import linalg as sLA
from typing import Union, List
from math import pi, log, pow
import mne

class Process():
    def __init__(self):
        # 定义采样率，题目文件中给出
        samp_rate = 250
        self.samp_rate = samp_rate
        # 选择导联编号
        self.select_channel = [50, 51, 52, 53, 54, 57, 58, 59]
        self.select_channel = [i - 1 for i in self.select_channel]
        self.n_ch = len(self.select_channel)
        # filterbank数量
        self.n_band = 2
        # 预处理滤波器设置
        self.filterB, self.filterA = self.__get_pre_filter(samp_rate)
        self.bpfilterB, self.bpfilterA = self.__get_bpfilter(samp_rate)
    def __get_pre_filter(self, samp_rate):
        fs = samp_rate
        f0 = 50
        q =  35
        b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
        return b, a

    def __get_bpfilter(self, samp_rate):
        passband1 = [3, 13]
        stopband1 = list(map(lambda x: x-2, passband1))

        passband2 = [80] * 10
        stopband2 = [90] * 10

        fs = samp_rate / 2
        bpfilterB = []
        bpfilterA = []
        for k in range(self.n_band):
            N, Wn = signal.ellipord([passband1[k] / fs, passband2[k] / fs], [stopband1[k] / fs, stopband2[k] / fs], 3, 40)
            b, a = signal.ellip(N, 0.5, 40, Wn, 'bandpass')
            bpfilterB.append(b)
            bpfilterA.append(a)

        return bpfilterB, bpfilterA

    def preprocess_fliter_bank(self, data):
        # 选择使用的导联
        data = data[self.select_channel, :]
        # data = (data - np.mean(data, axis=0))/np.std(data,axis=0)
        notchedData = signal.filtfilt(self.filterB, self.filterA, data)
        filteredData = []
        for k in range(self.n_band):
            _filteredData = signal.filtfilt(self.bpfilterB[k], self.bpfilterA[k], notchedData)

            filteredData.append(_filteredData)
        for i, _filteredData in enumerate(filteredData):
            def _get_data_delay(data, time: int):
                zero_column = np.zeros((self.n_ch, time))
                data_delay = np.concatenate([data[:, time:self.real_len], zero_column], axis=1)
                return np.concatenate([data, data_delay], axis=0)
            filteredData[i] = _get_data_delay(_filteredData, 2)

        # filteredData(ndarray):(n_bands, n_trials, n_chans, n_points)
        filteredData = np.array(filteredData)[:,np.newaxis, :, :]
        return filteredData

    def preprocess_withoutDelay_withoutFb(self, data):
        # 选择使用的导联
        data = data[self.select_channel, :]
        # data = (data - np.mean(data, axis=0))/np.std(data,axis=0)
        notchedData = signal.filtfilt(self.filterB, self.filterA, data)

        filteredData = signal.filtfilt(self.bpfilterB[0], self.bpfilterA[0], notchedData)

        return filteredData
    def preprocess_withDelay(self, data):
        # 选择使用的导联
        data = data[self.select_channel, :]
        # data = (data - np.mean(data, axis=0))/np.std(data,axis=0)
        notchedData = signal.filtfilt(self.filterB, self.filterA, data)

        filteredData = signal.filtfilt(self.bpfilterB[0], self.bpfilterA[0], notchedData)

        def _get_data_delay(data, time: int):
            zero_column = np.zeros((self.n_ch, time))
            data_delay = np.concatenate([data[:, time:1000], zero_column], axis=1)
            return np.concatenate([data, data_delay], axis=0)

        filteredData = _get_data_delay(filteredData,2)

        return filteredData
def __plot_freq(data:np.array,fs:int)->None:
    for i in range(len(data)):
        _data = data[i]
        _data = (_data - np.mean(_data)) / np.std(_data)
        n = len(_data)
        yf = np.fft.fft(_data)[:n//2]
        xf = np.fft.fftfreq(n, 1 / fs)[:n//2]
        plt.plot(xf, np.abs(yf)*2/n)
        plt.xlabel(f'Frequency (B:Hz) of {i}')
        plt.ylabel(f'Amplitude (A) of {i}')
        plt.grid()
        plt.show()


def __plot_eeg(data:Union[List,np.array]):
    data = np.array(data) if isinstance(data, list) else data
    print(data.shape)
    n_channel = data.shape[0]
    x = np.linspace(0, 4, data.shape[1])
    # 按行绘制每个子图
    fig, axs = plt.subplots(nrows=n_channel, ncols=1, figsize=(8, 10), sharex=True)
    for i in range(n_channel):
        axs[i].plot(x, data[i])
        axs[i].set_ylabel(f"Signal {i+1}")

    plt.xlabel("Time (s)")  # 设置共享的x轴标签
    plt.show()
def get_dataset_labels():
    labels = []
    for i in range(1, 3 + 1):
        for j in range(1, 4):
            m = np.load(f"/Users/yxy/Desktop/2023wrc/2023_bci_competition_frame-main_ssvep-main/Task/TestData/SSVEP/S00{i}/block00{j}.pkl", allow_pickle=True)
            data = m['data']
            trigger = data[-1,:]
            trigger_idx = np.where((trigger <= 40) & (trigger >= 1))[0]
            for t in trigger_idx:
                label = trigger[t]
                labels.append(label)
    labels = np.array(labels)
    return labels
# get_dataset_labels()

def get_template_info():
    template = np.load('./template.npy')
    print(template.shape)
    data = template[0][:, :1000]
    # for i in range(4):
    #     print(data[0][250*i:250*i+10])
    print(data[0][:250])
    print(data[1][:50])
    print(data[2][:50])
    print(data[3][:50])
    print(data[4][:50])
    print(data[4][250:300])
    print(data[4][500:550])

    x = np.linspace(0, 5, 1000)
    # 按行绘制每个子图
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(8, 10), sharex=True)
    for i in range(5):
        axs[i].plot(x, data[i])
        axs[i].set_ylabel(f"Signal {i+1}")

    plt.xlabel("Time (s)")  # 设置共享的x轴标签
    plt.show()

# get_template_info()

def get_my_template():

    sequence = np.load('./Sequence.npy')
    print(sequence.shape)

    # __plot_freq(sequence[:40],120)


    resampled_seq = scipy.signal.resample(sequence, 250, axis=1)
    print(resampled_seq.shape)

    raw_templates = np.load('./template.npy')
    templates = []
    for i, seq in enumerate(resampled_seq):
        seq = list(seq)
        cur_template = []
        # for j in range(10):
        #     # seq1 = (seq[-i:] + seq)[:250]
        #     seq2 = ([0 for _ in range(j)] * 2 + seq)[:250]
        #     cur_template.append(seq2*4)
        # __plot_freq(raw_templates[i], 250)
        # break
        for j, raw_template in enumerate(raw_templates[i]):

            cur_template.append(raw_template)

        # tmp = np.concatenate([np.zeros((1)), raw_templates[i][-1]], axis=0)[:1000]
        # cur_template.append(tmp)

        templates.append(cur_template)
    templates = np.array(templates)
    print(templates.shape)

    np.save('mytemplate.npy', np.array(templates))
#
# get_my_template()






def get_dataset_info():
    select_channel = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    select_channel = [i - 1 for i in select_channel]
    for i in range(1, 6):
        if i!=1:
            continue
        for j in range(1, 4):
            m = np.load(f"/Users/yxy/Desktop/2023wrc/dataset/S00{i}/block00{j}.pkl", allow_pickle=True)
            data = m['data']
            trigger = data[-1,:]
            trigger_idx = np.where((trigger <= 40) & (trigger >= 1))[0]
            trial_datas = [scipy.signal.resample(data[:,idx :idx+4000 ], 1000, axis=1) for idx in trigger_idx]
            trial_datas = np.array(trial_datas)
            for k,trial in enumerate(trial_datas):

                # if k!=43:
                #     continue
                trial_data = trial[select_channel,:]

                __plot_eeg(trial_data)
                # __plot_freq(trial_data, 250)
                # break
            break

# get_dataset_info()

def save_avg_template():
    process = Process()
    x_train = []
    y_train = []
    # select_channel = [50, 51, 52, 53, 54, 57, 58, 59]
    # select_channel = [i - 1 for i in select_channel]
    delay_time = int(0.10 * 250)
    cal_time = int(1.70 * 250)
    for i in range(1, 5 + 1):
        if i!=4:
            continue
        for j in range(1,  3 + 1):
            m = np.load(f"/Users/yxy/Desktop/2023wrc/dataset/S00{i}/block00{j}.pkl", allow_pickle=True)
            data = m['data']
            trigger = data[-1, :]
            trigger_idx = np.where((trigger <= 40) & (trigger >= 1))[0]
            trial_datas = [scipy.signal.resample(data[:, idx:idx + 4000], 1000, axis=1) for idx in trigger_idx]
            trial_datas = np.array(trial_datas)
            for x,y in zip(trial_datas,trigger_idx):
                x = process.preprocess_withoutDelay_withoutFb(x)
                def _get_data_delay(data, time: int):
                    zero_column = np.zeros((8, time))
                    data_delay = np.concatenate([data[:, time:1000], zero_column], axis=1)
                    return np.concatenate([data, data_delay], axis=0)
                # x_train.append(_get_data_delay(x, 1))
                x_train.append(x)
                y_train.append(trigger[y])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape)
    print(y_train.shape)
    print(np.unique(y_train).shape)
    np.save('./ecca_x_train.npy', x_train)
    np.save('./ecca_y_train.npy',y_train)
# #
# save_avg_template()
