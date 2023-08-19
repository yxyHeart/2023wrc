import numpy as np
from scipy.stats import kurtosis
from scipy import linalg
from numpy import linalg as nplinalg
import math
from collections import defaultdict
from scipy.special import softmax

'''
CCA3 是 CCA4 的改进版
主要变化是数据延迟可以选 1-10 之间k值最大的那个
结果是还不如之前的版本

'''
class CCA3:
    def __init__(self, target_template_set, n_band, n_ch, real_len):  # n_band:2, n_ch:10
        self.real_len = int(real_len)
        # 正余弦参考信号
        self.n_sti = len(target_template_set)  # n_sti: 40

        self.n_band = n_band
        self.n_ch = n_ch  # 10
        self.fb_coef = [a ** (-1.25) + 0.25 for a in range(1, n_band + 1)]
        # print(self.fb_coef)

        self.delay = 2
        self.cov_mat = np.zeros((self.delay * n_ch, self.delay * n_ch, n_band, 30))
        self.psf = np.zeros((self.delay * n_ch * 2, n_band, 30))  # (20,2,30)
        self.is_psf_ok = np.array([0] * 30)

        sig_len1 = 250
        sig_len2 = 500
        sig_len3 = 750
        sig_len4 = 1000
        self.QY1 = []
        self.QY2 = []
        self.QY3 = []
        self.QY4 = []
        self.QY5 = []
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 0:self.real_len]
            template = template.T
            template = template - template.mean(axis=0)
            [Q1, R1] = linalg.qr(template, mode='economic')
            self.QY1.append(Q1)
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 2:self.real_len+2]
            template = template.T
            template = template - template.mean(axis=0)
            [Q2, R2] = linalg.qr(template, mode='economic')
            self.QY2.append(Q2)
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 3:self.real_len+3]
            template = template.T
            template = template - template.mean(axis=0)
            [Q3, R3] = linalg.qr(template, mode='economic')
            self.QY3.append(Q3)
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 6:self.real_len+6]
            template = template.T
            template = template - template.mean(axis=0)
            [Q4, R4] = linalg.qr(template, mode='economic')
            self.QY4.append(Q4)
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 9:self.real_len+9]
            template = template.T
            template = template - template.mean(axis=0)
            [Q5, R5] = linalg.qr(template, mode='economic')
            self.QY5.append(Q5)   # (real_len,5)
        self.QY_all = {
            0: self.QY1,
            1: self.QY2,
            2: self.QY3,
            3: self.QY4,
            4: self.QY5,
        }

    def recognize(self, subdata, personID):  # subdata:(2,10,1000)
        subband_r1 = np.zeros((self.n_band, self.n_sti))  # (2,40)
        subband_r2 = np.zeros((self.n_band, self.n_sti))
        subband_r3 = np.zeros((self.n_band, self.n_sti))
        subband_r4 = np.zeros((self.n_band, self.n_sti))
        subband_r5 = np.zeros((self.n_band, self.n_sti))

        subband_rou_all = {
            0: [np.zeros((self.n_band, self.n_sti)) for _ in range(10)],
            1: [np.zeros((self.n_band, self.n_sti)) for _ in range(10)],
            2: [np.zeros((self.n_band, self.n_sti)) for _ in range(10)],
            3: [np.zeros((self.n_band, self.n_sti)) for _ in range(10)],
            4: [np.zeros((self.n_band, self.n_sti)) for _ in range(10)],
        }

        save_U1 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_U2 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_U3 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_U4 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_U5 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))

        save_U_all = {
            0: [np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band)) for _ in range(10)],
            1: [np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band)) for _ in range(10)],
            2: [np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band)) for _ in range(10)],
            3: [np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band)) for _ in range(10)],
            4: [np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band)) for _ in range(10)],
        }

        save_R1a = []
        save_R1b = []
        save_R1c = []
        save_R1d = []
        save_R1e = []

        save_R_all = {
            0: [[] for _ in range(10)],
            1: [[] for _ in range(10)],
            2: [[] for _ in range(10)],
            3: [[] for _ in range(10)],
            4: [[] for _ in range(10)],
        }


        def _get_data_delay(data,time:int):
            zero_column = np.zeros((self.n_ch, time))
            data_delay = np.concatenate([data[:, time:self.real_len], zero_column], axis=1)
            return np.concatenate([data, data_delay], axis=0)

        for i in range(5):
            for k in range(self.n_band):
                for delay in range(10):
                    data = _get_data_delay(subdata[k], delay+1)  # data:(20, 800)
                    data = data.T  # data: (800, 20)

                    W = self.psf[:, k, personID].reshape(self.n_ch * self.delay, 2)  # W: (20,1)

                    data = data - data.mean(axis=0)  #
                    [Q1a, R1a] = linalg.qr(data, mode='economic')  # Q1a: (800, 20) R1a:(20, 20)

                    save_R_all[i][delay].append(R1a)

                    if self.is_psf_ok[personID] == 1:
                        data1 = np.dot(data, W)
                        data1 = data1 - data1.mean(axis=0)  #
                        [Q11a, R11a] = linalg.qr(data1, mode='economic')  # Q11a:(800, 1)  R11:(1, 1)

                    for frequencyIndex in range(self.n_sti):
                        [svdU1, svdD1, svdV1] = nplinalg.svd(np.dot(Q1a.T, self.QY_all[i][frequencyIndex]))
                        save_U_all[i][delay][:, :, frequencyIndex, k] = svdU1  # svdU: (20, 20)

                        rho1a = 1 * svdD1[0]

                        if self.is_psf_ok[personID] == 1:
                            [svdU, svdD1, svdV] = nplinalg.svd(np.dot(Q11a.T, self.QY_all[i][frequencyIndex]))
                            rho2a = 1 * svdD1[0]
                        else:
                            rho2a = 0

                        subband_rou_all[i][delay][k][frequencyIndex] = (rho1a + rho2a) * self.fb_coef[k]

        rou = [None] * 5
        ten_delay_choose = [None] * 5
        # print(np.array(subband_rou_all[0]).shape)
        # print(subband_rou_all[0])
        for i in range(5):
            def f(x):
                x = softmax(x.sum(axis=0))
                return kurtosis(x)
            tmp_k = [f(x) for x in subband_rou_all[i]]
            # print(tmp_k)
            choose_idx = np.argmax(tmp_k)
            ten_delay_choose[i] = choose_idx
            # subband_rou_all[i].sort(reverse=True, key=lambda x:f(x))
            rou[i] = softmax(subband_rou_all[i][choose_idx].sum(axis=0))

        print(ten_delay_choose)
        tmp_rou_choose_flag = [False] * 5

        k_all = [None] * 5
        for i in range(5):
            k_all[i] = kurtosis(rou[i])

        res = [None] * 5
        for i in range(5):
            res[i] = int(np.argmax(rou[i]) + 1)


        k_avg = np.average(k_all)
        for i, k in enumerate(k_all):
            if k > k_avg+0.1:
                tmp_rou_choose_flag[i] = True
        print(tmp_rou_choose_flag)
        rou = np.sum([rou[i] if tmp_rou_choose_flag[i] else [0] * 40 for i in range(5)], axis=0)
        # rou = np.sum([tmp_r1, tmp_r2, tmp_r3, tmp_r4, tmp_r5], axis=0)
        ans = int(np.argmax(rou) + 1)

        ans_all = res
        print(ans_all, ans)
        update_flag = False
        idx = -1
        for i, ans_ in enumerate(ans_all):
            if ans_ == ans and k_all[i] == max(k_all) and k_all[i] >= 0:
                # print(k_all)
                update_flag = True
                idx = i
                break
        if update_flag:
            print('update!')
            save_U = save_U_all[idx][ten_delay_choose[idx]]  # (20, 20)
            save_R = save_R_all[idx][ten_delay_choose[idx]]  # (5, 5)
            for k in range(0, self.n_band):
                svdU = save_U[:, :, ans-1, k]
                wx = np.matmul(nplinalg.inv(save_R[k]), svdU)  # (20, 20)
                w0 = wx[:, 0:1]  # (20, 1)
                w0 = w0 / np.std(w0)
                w0 = w0.reshape(self.n_ch * self.delay, 1)
                self.cov_mat[:, :, k, personID] = self.cov_mat[:, :, k, personID] + np.dot(w0, w0.T)
                W, V = nplinalg.eig(self.cov_mat[:, :, k, personID])
                W = np.real(W)
                V = np.real(V)
                idx = [a for a in range(0, len(W)) if ~np.isinf(W[a])]
                W = W[idx]
                V = V[:, idx]
                idx = W.argsort()[::-1]
                V = V[:, idx]
                self.psf[:, k, personID] = np.array(V[:, 0:2]).reshape(-1)
                self.is_psf_ok[personID] = 1
        return ans,ans,-1,[]