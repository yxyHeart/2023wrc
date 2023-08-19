import numpy as np
from scipy.stats import kurtosis
from scipy import linalg
from numpy import linalg as nplinalg


class myCCA_v5e:
    def __init__(self, target_template_set, n_band, n_ch):
        # 正余弦参考信号

        self.n_sti = len(target_template_set)
        self.thres = 5.0
        self.thres_for_update = 8.0
        self.n_band = n_band
        self.n_ch = n_ch
        FB_coef0 = np.arange(0, n_band) + 1
        FB_coef0 = [a ** (-1.25) + 0.25 for a in FB_coef0]
        self.fb_coef = FB_coef0

        self.cov_mat = np.zeros((2 * n_ch, 2 * n_ch, n_band, 30))
        self.psf = np.zeros((2 * n_ch, n_band, 30))
        self.is_psf_ok = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.counter = 0
        # self.time_for_reset = 100000 # 38, 76, 114

        # self.ssvep_template = ssvep_template
        # QR decomposition of the reference
        sig_len = 800
        sig_len1 = 150
        sig_len2 = 300

        save_QY1 = []
        save_QY2 = []
        save_QY3 = []
        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, 0:sig_len]
            template = template.T
            template = template - template.mean(axis=0)
            [Q2, R2] = linalg.qr(template, mode='economic')
            save_QY1.append(Q2)

        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, sig_len1:sig_len]
            template = template.T
            template = template - template.mean(axis=0)
            [Q2, R2] = linalg.qr(template, mode='economic')
            save_QY2.append(Q2)

        for frequencyIndex in range(0, self.n_sti):
            template = target_template_set[frequencyIndex]
            template = template[:, sig_len2:sig_len]
            template = template.T
            template = template - template.mean(axis=0)
            [Q3, R3] = linalg.qr(template, mode='economic')
            save_QY3.append(Q3)

        self.QY1 = save_QY1
        self.QY2 = save_QY2
        self.QY3 = save_QY3

    # def reset(self):
    #     self.cov_mat = np.zeros((self.n_ch*2,self.n_ch*2,self.n_band))
    #     self.psf = np.zeros((self.n_ch*2,self.n_band))
    #     self.is_psf_ok = 0

    def recognize(self, subdata, personID):
        subband_r1 = np.zeros((self.n_band, self.n_sti))
        subband_r2 = np.zeros((self.n_band, self.n_sti))
        subband_r3 = np.zeros((self.n_band, self.n_sti))

        save_U1 = np.zeros((self.n_ch * 2, self.n_ch * 2, self.n_sti, self.n_band))
        save_U2 = np.zeros((self.n_ch * 2, self.n_ch * 2, self.n_sti, self.n_band))
        save_U3 = np.zeros((self.n_ch * 2, self.n_ch * 2, self.n_sti, self.n_band))
        save_R1a = []
        save_R1b = []
        save_R1c = []

        zero_column = np.zeros((self.n_ch, 1))
        sig_len = 800
        sig_len1 = 150
        sig_len2 = 300
        for k in range(0, self.n_band):

            data_delay = np.concatenate((subdata[k][:, 1:sig_len], zero_column), axis=1)
            data = np.concatenate((subdata[k], data_delay), axis=0)
            data = data.T

            W = self.psf[:, k, personID].reshape(data.shape[1], 1)

            data = data - data.mean(axis=0)  #
            [Q1a, R1a] = linalg.qr(data, mode='economic')

            data_ = data[sig_len1:sig_len, :]
            data_ = data_ - data_.mean(axis=0)  #
            [Q1b, R1b] = linalg.qr(data_, mode='economic')

            data__ = data[sig_len2:sig_len, :]
            data__ = data__ - data__.mean(axis=0)  #
            [Q1c, R1c] = linalg.qr(data__, mode='economic')

            save_R1a.append(R1a)
            save_R1b.append(R1b)
            save_R1c.append(R1c)

            if self.is_psf_ok[personID] == 1:
                data1 = np.dot(data, W)
                data1 = data1 - data1.mean(axis=0)  #
                [Q11a, R11] = linalg.qr(data1, mode='economic')

                data1 = np.dot(data_, W)
                data1 = data1 - data1.mean(axis=0)  #
                [Q11b, R11] = linalg.qr(data1, mode='economic')

                data1 = np.dot(data__, W)
                data1 = data1 - data1.mean(axis=0)  #
                [Q11c, R11] = linalg.qr(data1, mode='economic')

            for frequencyIndex in range(0, self.n_sti):
                # Q2 = self.QY1[frequencyIndex]
                # data_svd = np.dot(Q1a.T, self.QY1[frequencyIndex])
                [svdU, svdD1, svdV] = nplinalg.svd(np.dot(Q1a.T, self.QY1[frequencyIndex]))
                save_U1[:, :, frequencyIndex, k] = svdU
                [svdU, svdD2, svdV] = nplinalg.svd(np.dot(Q1b.T, self.QY2[frequencyIndex]))
                save_U2[:, :, frequencyIndex, k] = svdU
                [svdU, svdD3, svdV] = nplinalg.svd(np.dot(Q1c.T, self.QY3[frequencyIndex]))
                save_U3[:, :, frequencyIndex, k] = svdU

                rho1a = svdD1[0]
                rho1b = svdD2[0]
                rho1c = svdD3[0]

                if self.is_psf_ok[personID] == 1:
                    [svdU, svdD1, svdV] = nplinalg.svd(np.dot(Q11a.T, self.QY1[frequencyIndex]))
                    [svdU, svdD2, svdV] = nplinalg.svd(np.dot(Q11b.T, self.QY2[frequencyIndex]))
                    [svdU, svdD3, svdV] = nplinalg.svd(np.dot(Q11c.T, self.QY3[frequencyIndex]))
                    rho2a = svdD1[0]
                    rho2b = svdD2[0]
                    rho2c = svdD3[0]
                else:
                    rho2a = 0
                    rho2b = 0
                    rho2c = 0

                subband_r1[k, frequencyIndex] = (rho1a + rho2a) * self.fb_coef[k]
                subband_r2[k, frequencyIndex] = (rho1b + rho2b) * self.fb_coef[k]
                subband_r3[k, frequencyIndex] = (rho1c + rho2c) * self.fb_coef[k]

        tmp_r1 = subband_r1.sum(axis=0)
        tmp_r2 = subband_r2.sum(axis=0)
        tmp_r3 = subband_r3.sum(axis=0)
        k_val1 = kurtosis(tmp_r1, axis=0)
        k_val2 = kurtosis(tmp_r2, axis=0)
        k_val3 = kurtosis(tmp_r3, axis=0)
        result1 = np.argmax(tmp_r1)
        result2 = np.argmax(tmp_r2)
        result3 = np.argmax(tmp_r3)

        flag_det = 0
        if ((k_val1 >= k_val2) and (k_val1 >= k_val3)):
            if ((result1 == result2) and (result1 == result3)):
                result = result1
                flag_det = 1

            k_val = k_val1
            save_U = save_U1
            save_R = save_R1a
            # wx_pool = wx1_pool
        elif ((k_val2 >= k_val1) and (k_val2 >= k_val3)):
            if ((result2 == result3)):
                result = result2
                flag_det = 1
            k_val = k_val2
            save_U = save_U2
            save_R = save_R1b
            # wx_pool = wx2_pool
        elif ((k_val3 >= k_val1) and (k_val3 >= k_val2)):
            result = result3
            flag_det = 1
            k_val = k_val3
            save_U = save_U3
            save_R = save_R1c
        else:
            k_val = k_val1
            save_U = save_U1
            save_R = save_R1a

        if ((k_val > self.thres) and (flag_det == 1)):
            if k_val > self.thres_for_update:
                for k in range(0, self.n_band):
                    svdU = save_U[:, :, result, k]
                    wx = np.matmul(nplinalg.inv(save_R[k]), svdU)
                    w0 = wx[:, 0]
                    w0 = w0 / np.std(w0)
                    w0 = w0.reshape(data.shape[1], 1)
                    # w0=wx_pool[:,result,k].reshape(data.shape[1],1)
                    self.cov_mat[:, :, k, personID] = self.cov_mat[:, :, k, personID] + np.dot(w0, w0.T)
                    # A = self.cov_mat[:,:,k]
                    W, V = nplinalg.eig(self.cov_mat[:, :, k, personID])
                    W = np.real(W)
                    V = np.real(V)
                    idx = [a for a in range(0, len(W)) if ~np.isinf(W[a])]
                    W = W[idx]
                    V = V[:, idx]
                    idx = W.argsort()[::-1]
                    # r = W[idx]
                    V = V[:, idx]
                    self.psf[:, k, personID] = V[:, 0]
                    self.is_psf_ok[personID] = 1

            return result + 1
        else:
            return 0