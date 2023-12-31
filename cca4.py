import numpy as np
from scipy.stats import kurtosis
from scipy import linalg
from numpy import linalg as nplinalg
import math
from collections import defaultdict
from scipy.special import softmax


'''
CCA4 是 CCA2 的改进
CCA4 相较于 CCA2 的改进是：数据长度可以变化
'''
class CCA4:
    def __init__(self, target_template_set, n_band, n_ch, real_len):  # n_band:2, n_ch:10
        self.real_len = int(real_len)
        # 正余弦参考信号
        self.n_sti = len(target_template_set)  # n_sti: 40

        self.n_band = n_band
        self.n_ch = n_ch  # 10
        self.fb_coef = [a ** (-2) + 0.25 for a in range(1, n_band + 1)]


        self.delay = 2
        self.cov_mat = np.zeros((self.delay * n_ch, self.delay * n_ch, n_band, 30))
        self.psf = np.zeros((self.delay * n_ch, n_band, 30))  # (20,2,30)
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
        self.QY6 = []
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 0:self.real_len]
            template = template.T
            template = template - template.mean(axis=0)
            [Q1, R1] = linalg.qr(template, mode='economic')
            self.QY1.append(Q1)
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 1:self.real_len+1]
            template = template.T
            template = template - template.mean(axis=0)
            [Q2, R2] = linalg.qr(template, mode='economic')
            self.QY2.append(Q2)
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 2:self.real_len+2]
            template = template.T
            template = template - template.mean(axis=0)
            [Q3, R3] = linalg.qr(template, mode='economic')
            self.QY3.append(Q3)
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 3:self.real_len+3]
            template = template.T
            template = template - template.mean(axis=0)
            [Q4, R4] = linalg.qr(template, mode='economic')
            self.QY4.append(Q4)
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 4:self.real_len+4]
            template = template.T
            template = template - template.mean(axis=0)
            [Q5, R5] = linalg.qr(template, mode='economic')
            self.QY5.append(Q5)
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 5:self.real_len+5]
            template = template.T
            template = template - template.mean(axis=0)
            [Q6, R6] = linalg.qr(template, mode='economic')
            self.QY6.append(Q6)
        self.QY_all = {
            0: self.QY1,
            1: self.QY2,
            2: self.QY3,
            3: self.QY4,
            4: self.QY5,
            5: self.QY6
        }

    def recognize(self, subdata, personID):  # subdata:(2,10,1000)
        subband_r1 = np.zeros((self.n_band, self.n_sti))  # (2,40)
        subband_r2 = np.zeros((self.n_band, self.n_sti))
        subband_r3 = np.zeros((self.n_band, self.n_sti))
        subband_r4 = np.zeros((self.n_band, self.n_sti))
        subband_r5 = np.zeros((self.n_band, self.n_sti))
        subband_r6 = np.zeros((self.n_band, self.n_sti))

        subband_rou_all = {
            0: subband_r1,
            1: subband_r2,
            2: subband_r3,
            3: subband_r4,
            4: subband_r5,
            5: subband_r6

        }

        save_U1 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_U2 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_U3 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_U4 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_U5 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_U6 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))

        save_U_all = {
            0: save_U1,
            1: save_U2,
            2: save_U3,
            3: save_U4,
            4: save_U5,
            5: save_U6
        }

        save_R1a = []
        save_R1b = []
        save_R1c = []
        save_R1d = []
        save_R1e = []
        save_R1f = []

        save_R_all = {
            0: save_R1a,
            1: save_R1b,
            2: save_R1c,
            3: save_R1d,
            4: save_R1e,
            5: save_R1f
        }

        zero_column1 = np.zeros((self.n_ch, 1))
        zero_column2 = np.zeros((self.n_ch, 2))
        zero_column3 = np.zeros((self.n_ch, 3))
        zero_column4 = np.zeros((self.n_ch, 4))
        zero_column5 = np.zeros((self.n_ch, 5))
        zero_column6 = np.zeros((self.n_ch, 6))
        zero_column7 = np.zeros((self.n_ch, 7))
        zero_column8 = np.zeros((self.n_ch, 8))
        zero_column9 = np.zeros((self.n_ch, 9))
        zero_column10 = np.zeros((self.n_ch, 10))
        for i in range(6):
            for k in range(self.n_band):
                data_delay1 = np.concatenate((subdata[k][:, 1:self.real_len], zero_column1), axis=1)
                data_delay2 = np.concatenate((subdata[k][:, 2:self.real_len], zero_column2), axis=1)
                data_delay3 = np.concatenate((subdata[k][:, 3:self.real_len], zero_column3), axis=1)
                data_delay4 = np.concatenate((subdata[k][:, 4:self.real_len], zero_column4), axis=1)
                data_delay5 = np.concatenate((subdata[k][:, 5:self.real_len], zero_column5), axis=1)
                data_delay6 = np.concatenate((subdata[k][:, 6:self.real_len], zero_column6), axis=1)
                data_delay7 = np.concatenate((subdata[k][:, 7:self.real_len], zero_column7), axis=1)
                data_delay8 = np.concatenate((subdata[k][:, 8:self.real_len], zero_column8), axis=1)
                data_delay9 = np.concatenate((subdata[k][:, 9:self.real_len], zero_column9), axis=1)
                data_delay10 = np.concatenate((subdata[k][:, 10:self.real_len], zero_column10), axis=1)
                data = np.concatenate((subdata[k], data_delay6), axis=0)  # data:(20, 800)
                data = data.T  # data: (800, 20)

                W = self.psf[:, k, personID].reshape(self.n_ch * self.delay, 1)  # W: (20,1)

                data = data - data.mean(axis=0)  #
                [Q1a, R1a] = linalg.qr(data, mode='economic')  # Q1a: (800, 20) R1a:(20, 20)

                save_R_all[i].append(R1a)

                if self.is_psf_ok[personID] == 1:
                    data1 = np.dot(data, W)
                    data1 = data1 - data1.mean(axis=0)  #
                    [Q11a, R11a] = linalg.qr(data1, mode='economic')  # Q11a:(800, 1)  R11:(1, 1)


                for frequencyIndex in range(self.n_sti):
                    [svdU1, svdD1, svdV1] = nplinalg.svd(np.dot(Q1a.T, self.QY_all[i][frequencyIndex]))
                    save_U_all[i][:, :, frequencyIndex, k] = svdU1  # svdU: (20, 20)

                    rho1a = 1 * svdD1[0]

                    if self.is_psf_ok[personID] == 1:
                        [svdU, svdD1, svdV] = nplinalg.svd(np.dot(Q11a.T, self.QY_all[i][frequencyIndex]))
                        rho2a = 1 * svdD1[0]
                    else:
                        rho2a = 0

                    subband_rou_all[i][k, frequencyIndex] = (rho1a + rho2a) * self.fb_coef[k]

        tmp_r1 = softmax(subband_rou_all[0].sum(axis=0))
        tmp_r2 = softmax(subband_rou_all[1].sum(axis=0))
        tmp_r3 = softmax(subband_rou_all[2].sum(axis=0))
        tmp_r4 = softmax(subband_rou_all[3].sum(axis=0))
        tmp_r5 = softmax(subband_rou_all[4].sum(axis=0))
        tmp_r6 = softmax(subband_rou_all[5].sum(axis=0))

        # tmp_r1 = subband_rou_all[0].sum(axis=0)
        # tmp_r2 = subband_rou_all[1].sum(axis=0)
        # tmp_r3 = subband_rou_all[2].sum(axis=0)
        # tmp_r4 = subband_rou_all[3].sum(axis=0)

        tmp_rou_all = [tmp_r1, tmp_r2, tmp_r3, tmp_r4, tmp_r5, tmp_r6]
        tmp_rou_choose_flag = [False] * 6

        k1 = kurtosis(tmp_r1)
        k2 = kurtosis(tmp_r2)
        k3 = kurtosis(tmp_r3)
        k4 = kurtosis(tmp_r4)
        k5 = kurtosis(tmp_r5)
        k6 = kurtosis(tmp_r6)

        k_all = [k1, k2, k3, k4, k5, k6]
        res1 = int(np.argmax(tmp_r1) + 1)
        res2 = int(np.argmax(tmp_r2) + 1)
        res3 = int(np.argmax(tmp_r3) + 1)
        res4 = int(np.argmax(tmp_r4) + 1)
        res5 = int(np.argmax(tmp_r5) + 1)
        res6 = int(np.argmax(tmp_r6) + 1)


        k_avg = np.average(k_all)
        for i, k in enumerate(k_all):
            # if k > k_avg:
            #     tmp_rou_choose_flag[i] = True
            tmp_rou_choose_flag[i] = True

        rou = np.sum([tmp_rou_all[i] if tmp_rou_choose_flag[i] else [0] * 40 for i in range(5)], axis=0)

        ans = int(np.argmax(rou) + 1)

        ans_all = [res1, res2, res3, res4, res5, res6]
        print(ans_all, ans)
        update_flag = False
        idx = -1
        for i, ans_ in enumerate(ans_all):
            if ans_ == ans and k_all[i] == max(k_all) and k_all[i] >= 2:
                update_flag = True
                idx = i
                break
        if update_flag:
            print('update!')
            save_U = save_U_all[idx]
            save_R = save_R_all[idx]
            for k in range(0, self.n_band):
                svdU = save_U[:, :, ans-1, k]
                wx = np.matmul(nplinalg.inv(save_R[k]), svdU)
                w0 = wx[:, 0]
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
                self.psf[:, k, personID] = V[:, 0]
                self.is_psf_ok[personID] = 1
        return rou, ans