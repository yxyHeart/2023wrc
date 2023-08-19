import numpy as np
from scipy.stats import kurtosis
from scipy import linalg
from numpy import linalg as nplinalg
import math
from collections import defaultdict
from scipy.special import softmax

'''
CCA5 是结合 CCA 和 FBCCA 的版本 (还没写）
'''
class CCA5:
    def __init__(self, target_template_set, n_band, n_ch, real_len):  # n_band:2, n_ch:10
        self.real_len = int(real_len)
        # 正余弦参考信号
        self.n_sti = len(target_template_set)  # n_sti: 40

        self.n_band = n_band
        self.n_ch = n_ch  # 10
        self.fb_coef = [a ** (-1.25) + 0.25 for a in range(1, n_band + 1)]


        self.delay = 2
        self.cov_mat = np.zeros((self.delay * n_ch, self.delay * n_ch, n_band, 30))
        self.psf = np.zeros((self.delay * n_ch, n_band, 30))  # (20,2,30)
        self.is_psf_ok = np.array([0] * 30)


        self.QY1 = []

        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 0:self.real_len]
            template = template.T
            template = template - template.mean(axis=0)
            [Q1, R1] = linalg.qr(template, mode='economic')
            self.QY1.append(Q1)


    def recognize(self, subdata, personID):  # subdata:(2,10,1000)
        subband_r1 = np.zeros((self.n_band, self.n_sti))  # (2,40)
        save_U1 = np.zeros((self.n_ch * self.delay, self.n_ch * self.delay, self.n_sti, self.n_band))
        save_R1a = []

        def _get_data_delay(data, time: int):
            zero_column = np.zeros((self.n_ch, time))
            data_delay = np.concatenate([data[:, time:self.real_len], zero_column], axis=1)
            return np.concatenate([data, data_delay], axis=0)
        for k in range(self.n_band):
            data = _get_data_delay(subdata[k], 1)  # data:(20, 800)
            data = data.T  # data: (800, 20)

            W = self.psf[:, k, personID].reshape(self.n_ch * self.delay, 1)  # W: (20,1)

            data = data - data.mean(axis=0)  #
            [Q1a, R1a] = linalg.qr(data, mode='economic')  # Q1a: (800, 20) R1a:(20, 20)

            save_R1a.append(R1a)

            if self.is_psf_ok[personID] == 1:
                data1 = np.dot(data, W)
                data1 = data1 - data1.mean(axis=0)  #
                [Q11a, R11a] = linalg.qr(data1, mode='economic')  # Q11a:(800, 1)  R11:(1, 1)


            for frequencyIndex in range(self.n_sti):
                [svdU1, svdD1, svdV1] = nplinalg.svd(np.dot(Q1a.T, self.QY1[frequencyIndex]))
                save_U1[:, :, frequencyIndex, k] = svdU1  # svdU: (20, 20)

                rho1a = 1 * svdD1[0]

                if self.is_psf_ok[personID] == 1:
                    [svdU, svdD1, svdV] = nplinalg.svd(np.dot(Q11a.T, self.QY1[frequencyIndex]))
                    rho2a = 1 * svdD1[0]
                else:
                    rho2a = 0

                subband_r1[k, frequencyIndex] = (rho1a + rho2a) * self.fb_coef[k]
        #
        # tmp_r1 = softmax(subband_r1.sum(axis=0))
        tmp_r1 = subband_r1.sum(axis=0)
        tmp_r1 = softmax(tmp_r1)
        k1 = kurtosis(tmp_r1)
        # print(tmp_r1,k1)
        res1 = int(np.argmax(tmp_r1) + 1)

        print(res1)
        if k1>=999:
            print('update!')
            save_U = save_U1
            save_R = save_R1a
            for k in range(0, self.n_band):
                svdU = save_U[:, :, res1-1, k]
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
        return res1,res1,-1,[]