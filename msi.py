# Designer:Yudong Pan
# Coder:God's hand
# Time:2023/4/13 19:24
try:
    from Algorithm.impl import utils
except:
    import utils
import numpy as np
from numpy import ndarray

class MSI():
    def fit(self, X_train:ndarray, Y_train:ndarray):
        '''
        Args:
            X_train (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
            y_train (ndarray): (n_events,). Labels for X_train.

        '''
        self.X_train = X_train
        self.Y_train = Y_train
        event_type = np.unique(self.Y_train)
        self.train_info = {
            'event_type':event_type,
            'n_events':len(event_type),
            'n_points':self.X_train.shape[-1]
        }
        return self

    def predict(self, X_test:ndarray):
        """Using MSI to predict test data.

        Args:
            X_test (ndarray): (test_trials, n_chans, n_points).
                Test dataset. test_trials could be 1 if necessary.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']
        event_type = self.train_info['event_type']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            self.rou[nte] = self.__find_Synchronization_Index(
                X = X_test[nte],
                Y = self.X_train
            )
            self.y_predict[nte] = event_type[np.argmax(self.rou[nte,:])]

        return self.rou, self.y_predict
    def __find_Synchronization_Index(self, X, Y):
        num_freq = Y.shape[0]
        num_harm = Y.shape[1]
        result = np.zeros(num_freq)
        self.Nc = X.shape[0]
        self.T = X.shape[1]
        for freq_idx in range(0, num_freq):
            y = Y[freq_idx]
            X = X[:] - np.mean(X).repeat(self.T * self.Nc).reshape(self.Nc, self.T)
            X = X[:] / np.std(X).repeat(self.T * self.Nc).reshape(self.Nc, self.T)

            y = y[:] - np.mean(y).repeat(self.T * num_harm).reshape(num_harm, self.T)
            y = y[:] / np.std(y).repeat(self.T * num_harm).reshape(num_harm, self.T)

            c11 = (1 / self.T) * (np.dot(X, X.T))
            c22 = (1 / self.T) * (np.dot(y, y.T))
            c12 = (1 / self.T) * (np.dot(X, y.T))
            c21 = c12.T

            C_up = np.column_stack([c11, c12])
            C_down = np.column_stack([c21, c22])
            C = np.row_stack([C_up, C_down])

            # print("c11:", c11)
            # print("c22:", c22)

            v1, Q1 = np.linalg.eig(c11)
            v2, Q2 = np.linalg.eig(c22)
            V1 = np.diag(v1 ** (-0.5))
            V2 = np.diag(v2 ** (-0.5))

            C11 = np.dot(np.dot(Q1, V1.T), np.linalg.inv(Q1))
            C22 = np.dot(np.dot(Q2, V2.T), np.linalg.inv(Q2))

            # print("Q1 * Q1^(-1):", np.dot(Q1, np.linalg.inv(Q1)))
            # print("Q2 * Q2^(-1):", np.dot(Q2, np.linalg.inv(Q2)))

            U_up = np.column_stack([C11, np.zeros((self.Nc, num_harm))])
            U_down = np.column_stack([np.zeros((y.shape[0], self.Nc)), C22])
            U = np.row_stack([U_up, U_down])
            R = np.dot(np.dot(U, C), U.T)

            eig_val, _ = np.linalg.eig(R)
            # print("eig_val:", eig_val, eig_val.shape)
            E = eig_val / np.sum(eig_val)
            S = 1 + np.sum(E * np.log(E)) / np.log(self.Nc + num_harm)
            result[freq_idx] = S

        return result

class FB_MSI():
    def fit(self, X_train:ndarray, Y_train:ndarray):
        '''
        Args:
            X_train (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
            y_train (ndarray): (n_events,). Labels for X_train.

        '''
        self.X_train = X_train
        self.Y_train = Y_train
        event_type = np.unique(self.Y_train)
        self.train_info = {
            'event_type':event_type,
            'n_events':len(event_type),
            'n_points':self.X_train.shape[-1]
        }
        return self

    def predict(self,X_test):
        """Using filter-bank CCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_bands, test_trials, n_chans, n_points).
                Test dataset. test_trials could be 1 if necessary.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_bands = X_test.shape[0]
        n_test = X_test.shape[1]
        # apply CCA().predict() in each sub-band
        self.sub_models = [[] for nb in range(n_bands)]
        self.fb_rou = [[] for nb in range(n_bands)]
        self.fb_y_predict = [[] for nb in range(n_bands)]
        for nb in range(n_bands):
            self.sub_models[nb] = MSI()
            self.sub_models[nb].fit(
                X_train=self.X_train,
                Y_train=self.Y_train
            )
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
            self.fb_rou[nb], self.fb_y_predict[nb] = fb_results[0], fb_results[1]

        # integration of multi-bands results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict

if __name__ == '__main__':
    # msi = MSI()
    # x_train = np.random.randn(40,5,400)
    # y_train = np.array(list(range(40)))
    # x_test = np.random.randn(1,8,400)
    #
    # msi.fit(x_train,y_train)
    # rou,result = msi.predict(x_test)
    # print(result)

    fbmsi = FB_MSI()
    x_train = np.random.randn(40,5,400)
    y_train = np.array(list(range(40)))
    x_test = np.random.randn(5,1,8,400)

    fbmsi.fit(x_train,y_train)
    rou,result = fbmsi.predict(x_test)
    print(rou)
    print(result)