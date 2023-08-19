from Algorithm.AlgorithmInterface import AlgorithmInterface
from Algorithm.impl.cca import FB_CCA,FB_ECCA
from Algorithm.impl.trca import FB_TRCA
from Algorithm.impl.msi import FB_MSI
from Algorithm.impl.cca4 import CCA4
from scipy import signal
import numpy as np
import math
import os
from Algorithm.impl import analysis
from decimal import Decimal
import sys
import mne
from scipy.special import softmax
import scipy
import warnings
warnings.filterwarnings("ignore")


# 一次只有40ms数据也就是10个采样点的数据
class AlgorithmImplement(AlgorithmInterface):
    def __init__(self):
        super().__init__()
        # 定义采样率，题目文件中给出
        samp_rate = 250
        self.samp_rate = samp_rate
        # 选择导联编号
        self.select_channel = [50, 51, 52, 53, 56, 57, 58, 59]
        self.select_channel = [i - 1 for i in self.select_channel]
        self.n_ch = len(self.select_channel)
        # trial开始trigger，题目说明中给出
        self.trial_start_trig = 240
        self.trial_end_trig = 241
        # 计算时间
        cal_time = Decimal(str(1.20))
        # 计算偏移时间（s）
        offset_time = Decimal(str(0.10))

        self.offset_len = math.floor(offset_time * samp_rate)
        # 计算长度
        self.cal_len = int(cal_time * samp_rate)
        # 偏移长度, python 有浮点数误差
        self.real_len = int(((cal_time - offset_time) * samp_rate))
        # filterbank数量
        self.n_band = 2
        # 预处理滤波器设置
        self.filterB, self.filterA = self.__get_pre_filter(samp_rate)
        self.bpfilterB, self.bpfilterA = self.__get_bpfilter(samp_rate)
        # 正余弦参考信号
        target_template_set = np.array(self.__get_target_template_set())
        # 初始化fbcca算法
        y_train = list(range(40))
        self.method1 = FB_CCA()
        self.method1.fit(target_template_set, y_train)

        # 初始化fbecca算法
        x_train = np.load(os.path.dirname(__file__) + '/ecca_x_train.npy')[:, :, self.offset_len:self.cal_len]
        y_train = np.load(os.path.dirname(__file__) + '/ecca_y_train.npy')
        sine_template = target_template_set
        self.method2 = FB_ECCA()
        self.method2.fit(x_train, y_train, sine_template)

        # 初始化fbmsi算法
        self.method3 = FB_MSI()
        y_train = list(range(40))
        self.method3.fit(target_template_set,y_train)

        self.ans = [[] for _ in range(30)]

        # 初始化qr分解版本的fbcca算法
        self.method4 = CCA4(target_template_set, self.n_band, self.n_ch, self.real_len)


    def run(self):
        # 是否停止标签
        end_flag = False
        # 是否进入计算模式标签
        cal_flag = False
        while not end_flag:
            data_model = self.comm_proxy.get_data()
            if data_model is None:
                continue
            data_model.data = np.array(data_model.data)
            if not cal_flag:
                # 非计算模式，则进行事件检测
                cal_flag = self.__idle_proc(data_model)
            else:
                # 计算模式，则进行处理
                cal_flag, result= self.__cal_proc(data_model)
                # 如果有结果，则进行报告
                if result is not None:
                    res = int(result[0])
                    print(res)
                    personID = int(data_model.subject_id)
                    self.ans[personID-1].append(res)
                    self.comm_proxy.report(res)
                    # 清空缓存
                    self.__clear_cache()
            end_flag = data_model.finish_flag
        #
        self.__cal_acc()

    def __idle_proc(self, data_model):

        # 脑电数据+trigger
        data = data_model.data
        # 获取trigger导
        trigger = data[-1, :]
        # trial开始类型的trigger所在位置的索引
        trigger_idx = np.where(trigger == self.trial_start_trig)[0]
        # 脑电数据
        eeg_data = data[0: -1, :]
        if len(trigger_idx) > 0:
            # 有trial开始trigger则进行计算
            cal_flag = True
            trial_start_trig_pos = trigger_idx[0]
            # 从trial开始的位置拼接数据
            self.cache_data = eeg_data[:, trial_start_trig_pos: eeg_data.shape[1]]
        else:
            # 没有trial开始trigger则
            cal_flag = False
            self.__clear_cache()
        return cal_flag

    def __cal_proc(self, data_model):
        # 获取当前被试id
        personID = int(data_model.subject_id)
        # 脑电数据+trigger
        data = data_model.data
        # 获取trigger导
        trigger = data[-1, :]
        # trial开始类型的trigger所在位置的索引
        trigger_idx = np.where(trigger == self.trial_end_trig)[0]
        # 获取脑电数据
        eeg_data = data[0: -1, :]
        # 如果trigger为空，表示依然在当前试次中，根据数据长度判断是否计算
        if len(trigger_idx) == 0:
            # 当已缓存的数据大于等于所需要使用的计算数据时，进行计算
            if self.cache_data.shape[1] >= self.cal_len :
                # 获取所需计算长度的数据
                self.cache_data = self.cache_data[:, 0: self.cal_len]
                # 考虑偏移量
                use_data = self.cache_data[:, self.offset_len: self.cache_data.shape[1]]
                # 滤波处理
                use_data = self.__preprocess(use_data)

                rou, result = self.method4.recognize(use_data.squeeze(), personID)


                result = [result]
                cal_flag = False

            else:
                # 反之继续采集数据
                self.cache_data = np.append(self.cache_data, eeg_data, axis=1)
                result = None
                cal_flag = True
        # 试次已经结束,需要强制结束计算
        else:
            # trial结束trigger的位置
            trial_end_trig_pos = trigger_idx[0]
            # 如果拼接该数据包中部分的数据后，可以满足所需要的计算长度，则拼接数据达到所需要的计算长度
            # 如果拼接完该trial的所有数据后仍无法满足所需要的数据长度，则只能使用该trial的全部数据进行计算
            use_len = min(trial_end_trig_pos, self.cal_len - self.cache_data.shape[1])
            self.cache_data = np.append(self.cache_data, eeg_data[:, 0: int(use_len)], axis=1)
            # 考虑偏移量
            use_data = self.cache_data[:, self.offset_len: self.cache_data.shape[1]]
            # 滤波处理
            use_data = self.__preprocess(use_data)
            # 开始计算
            rou, result = self.method.predict(use_data)

            # 开始新试次的计算模式
            cal_flag = False
            # 清除缓存的数据
            self.__clear_cache()
        return cal_flag, result

    def __get_pre_filter(self, samp_rate):
        fs = samp_rate
        f0 = 50
        q =  35
        b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
        return b, a

    def __get_bpfilter(self, samp_rate):
        passband1 = [6, 13, 23, 33, 43]
        stopband1 = list(map(lambda x: x-2, passband1))

        passband2 = [35] * 10
        stopband2 = [45] * 10

        fs = samp_rate / 2
        bpfilterB = []
        bpfilterA = []
        for k in range(self.n_band):
            N, Wn = signal.ellipord([passband1[k] / fs, passband2[k] / fs], [stopband1[k] / fs, stopband2[k] / fs], 3, 40)
            b, a = signal.ellip(N, 0.5, 40, Wn, 'bandpass')
            bpfilterB.append(b)
            bpfilterA.append(a)

        return bpfilterB, bpfilterA

    def __clear_cache(self):
        self.cache_data = None

    def __preprocess(self, data):
        # 选择使用的导联
        data = data[self.select_channel, :]
        # data = (data - np.mean(data, axis=0))/np.std(data,axis=0)
        notchedData = signal.filtfilt(self.filterB, self.filterA, data)
        filteredData = []
        for k in range(self.n_band):
            _filteredData = signal.filtfilt(self.bpfilterB[k], self.bpfilterA[k], notchedData)

            filteredData.append(_filteredData)
        for i, _filteredData in enumerate(filteredData):
            continue
            def _get_data_delay(data, time: int):
                zero_column = np.zeros((self.n_ch, time))
                data_delay = np.concatenate([data[:, time:self.real_len], zero_column], axis=1)
                return np.concatenate([data, data_delay], axis=0)

            filteredData[i] = _get_data_delay(_filteredData, 2)

        # filteredData(ndarray):(n_bands, n_trials, n_chans, n_points)
        filteredData = np.array(filteredData)[:,np.newaxis, :, :]
        return filteredData

    def __get_target_template_set(self):
        target_template_set = []

        filename = os.path.dirname(__file__)+'/mytemplate.npy'
        seq = np.load(filename)
        # print(seq.shape[0])
        for i in range(seq.shape[0]):
            target_template_set.append(seq[i])

        return target_template_set

    def __cal_acc(self):
        labels_all = np.array(analysis.get_dataset_labels())
        self.ans = np.array(self.ans[:3])
        for i in range(3):
            cur_labels = labels_all[i*135:(i+1)*135]
            true_cnt = 0
            for j in range(135):
                if cur_labels[j] == self.ans[i][j]:
                    true_cnt += 1
            cnt = np.sum(cur_labels == self.ans[i])
            acc = cnt / (len(self.ans[i]))
            print(f'S00{i+1}: acc:{acc} true_cnt:{true_cnt}')



