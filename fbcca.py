import numpy as np
from scipy.stats import kurtosis
from scipy import linalg
from numpy import linalg as nplinalg
import math
from collections import defaultdict
from scipy.special import softmax


class CCA:
    def __init__(self, target_template_set, n_band, n_ch):  # n_band:2, n_ch:10
        # 正余弦参考信号
        self.n_sti = len(target_template_set)  # n_sti: 40

        self.n_band = n_band
        self.n_ch = n_ch  # 10
        self.fb_coef = [a ** (-1.25) + 0.25 for a in range(1, n_band + 1)]
        # print(self.fb_coef)

        # QR decomposition of the reference
        save_QY1 = []
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 0:1000]
            template = template.T
            template = template - template.mean(axis=0)
            [Q1, R1] = linalg.qr(template)
            save_QY1.append(Q1)

        self.QY1 = save_QY1  # (40, 800, 10)

    def epsilon(self, p):
        a = p[0] - p[1]
        p = np.array(p)
        b = p.sum() - p.shape[0] * np.log(np.exp(p).sum())
        return a / b

    def recognize(self, subdata, personID, real_len):  # subdata:(2,10,1000)
        # print(np.array(subdata).shape,real_len)
        real_len = int(real_len)
        subdata = np.array(subdata)
        subband_r1 = np.zeros((self.n_band, self.n_sti))  # (2,40)
        zero_column = np.zeros((self.n_ch, 2))

        for k in range(0, self.n_band):
            data_delay = np.concatenate((subdata[k][:, 2:real_len], zero_column), axis=1)
            data = np.concatenate((subdata[k], data_delay), axis=0)  # data:(20, 800)
            data = data.T  # data: (800, 20)

            data = data - data.mean(axis=0)  #
            [Q1a, R1a] = linalg.qr(data, mode='economic')  # Q1a: (800, 20) R1a:(20, 20)

            for frequencyIndex in range(0, self.n_sti):
                [svdU1, svdD1, svdV1] = nplinalg.svd(np.dot(Q1a.T, self.QY1[frequencyIndex][:real_len]))
                rho1a = svdD1[0]
                subband_r1[k, frequencyIndex] = (rho1a) * self.fb_coef[k]

        tmp_r1 = subband_r1.sum(axis=0)

        conf = kurtosis(tmp_r1, axis=0)
        result1 = int(np.argmax(tmp_r1) + 1)
        # if conf < 0.10:
        #     return 0, result1, conf, tmp_r1
        return result1, result1, conf, tmp_r1