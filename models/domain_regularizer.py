# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:53:15 2016

@author: zellinger
"""

from keras import backend as K
from keras.regularizers import Regularizer
import theano.tensor as T
import tensorflow as tf
import math
import numpy as np
tf_session = K.get_session()


class DomainRegularizer(Regularizer):
    def __init__(self,l=1,name='mmd',beta=1.0):
        self.uses_learning_phase = 1
        self.l=l
        self.name=name
        self.beta=beta

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        regularizer_loss = loss
        sim = 0
        if len(self.layer.inbound_nodes)>1:
            if self.name=='mmd':
                sim = self.mmd(self.layer.get_output_at(0),
                               self.layer.get_output_at(1),
                               self.beta)
            elif self.name=='mmd5':
                sim = self.mmdK(self.layer.get_output_at(0),
                                self.layer.get_output_at(1),
                                5)
            elif self.name=='mmatch':
                sim = self.mmatch(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  5)
            elif self.name=='mmatchK':
                sim = self.mmatch(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='SMD_D1':
                sim = self.SMD_D1(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='SMD_D1_HAT':
                sim = self.SMD_D1_HAT(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='SMD_D2':
                sim = self.SMD_D2(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='SMD_D2_HAT':
                sim = self.SMD_D2_HAT(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='DWMD':
                sim = self.DWMD(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='DWMD1':
                sim = self.DWMD1(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='DWMD2':
                sim = self.DWMD2(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='DWMD3':
                sim = self.DWMD3(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='DWMD4':
                sim = self.DWMD4(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='DWMD5':
                sim = self.DWMD5(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='DWMD6':
                sim = self.DWMD6(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='DWMD7':
                sim = self.DWMD7(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            else:
                print('ERROR: Regularizer not supported.')

        add_loss = K.switch(K.equal(len(self.layer.inbound_nodes),2),sim,K.update_add(K.variable(0),0))
        # print(len(self.layer.inbound_nodes))
        # add_loss = tf.cond(K.equal(len(self.layer.inbound_nodes), 2), lambda :sim, )
        # if K.equal(len(self.layer.inbound_nodes),2):
        #     add_loss = sim
        # else:
        #     add_loss = 0
        regularizer_loss += self.l*add_loss

        return K.in_train_phase(regularizer_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l': float(self.l)}
        
    def mmd(self, x1, x2, beta):
        x1x1 = self.gaussian_kernel(x1, x1, beta)
        x1x2 = self.gaussian_kernel(x1, x2, beta)
        x2x2 = self.gaussian_kernel(x2, x2, beta)
        diff = K.mean(x1x1) - 2 * K.mean(x1x2) + K.mean(x2x2)
        return diff

    def mmatch(self, x1, x2, n_moments):
        mx1 = K.mean(x1, axis=0)
        mx2 = K.mean(x2, axis=0)
        # mx1 = x1.mean(0)
        # mx2 = x2.mean(0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1,mx2)
        scms = dm
        for i in range(n_moments-1):
            scms+=self.scm(sx1,sx2,i+2)
        return scms
        
    def mmdK(self, x1, x2, n_moments):
        mx1 = x1.mean(0)
        mx2 = x2.mean(0)
        s1=mx1
        s2=mx2
        for i in range(n_moments-1):
            s1+=(x1**T.cast(i+2,'int32')).mean(0)
            s2+=(x2**T.cast(i+2,'int32')).mean(0)
        return K.sqrt(K.sum((s1-s2)**2))

    def gaussian_kernel(self, x1, x2, beta = 2.0):
        return K.exp(-beta * K.sum(K.pow(x1 -x2, 2)))

    def scm(self, sx1, sx2, k):
        ss1 = K.mean(sx1**K.cast(k, 'float32'), axis=0)
        ss2 = K.mean(sx2**K.cast(k, 'float32'), axis=0)

        # K.cast()
        # ss1 = (sx1**T.cast(k,'int32')).mean(0)
        # ss2 = (sx2**T.cast(k,'int32')).mean(0)
        return self.matchnorm(ss1,ss2)
        
    def matchnorm(self, x1, x2):
        return K.sqrt(K.sum((x1-x2)**2))

    def SMD_D1(self, x1, x2, k):
        # k是几阶距
        t = 3
        beta = 2
        order = 5
        gamma = 1
        sing_item = 0
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = K.pow(K.abs(EXs_n - EXt_n), beta)
            sing_item += math.pow(1 / t, math.sqrt(n)) * (numerator / (numerator + 0.05))
        return K.sum(sing_item)

    def SMD_D1_HAT(self, x1, x2, k):
        t = 3
        beta = 2
        order = 5
        gamma = 1
        sing_item = 0
        EXs = K.mean(x1, axis=0)
        EXt = K.mean(x2, axis=0)
        DXs_std = K.std(x1, axis=0)
        DXt_std = K.std(x2, axis=0)
        for n in range(1, (order + 1)):
            EXs_n_hat = K.mean(K.pow((x1 - EXs) / DXs_std, n), axis=0)
            EXt_n_hat = K.mean(K.pow((x2 - EXt) / DXt_std, n), axis=0)
            numerator = K.pow(K.abs(EXs_n_hat - EXt_n_hat), beta)
            sing_item += math.pow(1 / t, n) * (numerator / (numerator + 0.05))
        return K.sum(sing_item)

    def SMD_D2(self, x1, x2, k):
        t = 6
        beta = 2
        order = 5
        gamma = 5
        sing_item = 0
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = gamma * K.exp(K.abs(EXs_n - EXt_n))
            sing_item += math.pow(1 / t, math.sqrt(n)) * (numerator / (numerator + 0.05))
        return K.sum(sing_item)
        
    def SMD_D2_HAT(self, x1, x2, k):
        t = 3
        beta = 2
        order = 5
        gamma = 1
        sing_item = 0
        EXs = K.mean(x1, axis=0)
        EXt = K.mean(x2, axis=0)
        DXs_std = K.std(x1, axis=0)
        DXt_std = K.std(x2, axis=0)
        for n in range(1, (order + 1)):
            EXs_n_hat = K.mean(K.pow((x1 - EXs) / DXs_std, n), axis=0)
            EXt_n_hat = K.mean(K.pow((x2 - EXt) / DXt_std, n), axis=0)
            numerator = gamma * K.exp(K.abs(EXs_n_hat - EXt_n_hat))
            sing_item += math.pow(1 / t, n) * (numerator / (numerator + 0.05))
        return K.sum(sing_item)

    def DWMD(self, x1, x2, k):
        lamb = 0.1
        t = 3
        beta = 1
        order = 5
        gamma = 3
        sing_item = 0
        x1_tmu = np.load("Xs.npy")
        x2_tmu = np.load("Xt.npy")
        x_tmux = np.abs(x1_tmu - x2_tmu).ravel().astype(np.float32)
        x_tmux = x_tmux / np.max(x_tmux)
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = gamma * K.pow(K.abs(EXs_n - EXt_n), beta)
            temp = K.exp(-lamb * math.sqrt(n) / x_tmux) * numerator / (numerator + 0.05)
            sing_item += temp
        return K.sum(sing_item)

    def DWMD1(self, x1, x2, k):
        lamb = 0.1
        t = 3
        beta = 1
        order = 5
        gamma = 1
        sing_item = 0
        x1_tmu = np.load("Xs.npy")
        x2_tmu = np.load("Xt.npy")
        x_tmux = np.abs(x1_tmu - x2_tmu).ravel().astype(np.float32)
        x_tmux = x_tmux / np.max(x_tmux)
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = gamma * K.exp(K.abs(EXs_n - EXt_n))
            temp = K.exp(-lamb * math.sqrt(n) / x_tmux) * numerator / (numerator + 0.05)
            sing_item += temp
        return K.sum(sing_item)

    def DWMD2(self, x1, x2, k):
        lamb = 0.1
        t = 3
        beta = 1
        order = 5
        gamma = 1
        gamma_1 = 5
        sing_item = 0
        x1_tmu = np.load("Xs.npy")
        x2_tmu = np.load("Xt.npy")
        x_tmux = np.abs(x1_tmu - x2_tmu).ravel().astype(np.float32)
        x_tmux = x_tmux / np.max(x_tmux)
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = gamma * K.pow(K.abs(EXs_n - EXt_n), beta)
            temp = gamma * K.exp(-1 / (1 - K.pow(K.abs(x_tmux / math.sqrt(n) - 1), 2))) * numerator / (numerator + 0.05)
            sing_item += temp
        return K.sum(sing_item)

    def DWMD3(self, x1, x2, k):
        lamb = 0.1
        t = 3
        beta = 1
        order = 5
        gamma = 5
        gamma_1 = 3
        sing_item = 0
        x1_tmu = np.load("Xs.npy")
        x2_tmu = np.load("Xt.npy")
        x_tmux = np.abs(x1_tmu - x2_tmu).ravel().astype(np.float32)
        x_tmux = x_tmux / np.max(x_tmux)
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = gamma * K.pow(K.abs(EXs_n - EXt_n), beta)
            temp = gamma * K.exp(-1 / (1 - K.pow(K.abs(x_tmux / math.sqrt(n) - 1), 2))) * numerator / (numerator + 0.05)
            sing_item += temp
        return K.sum(sing_item)
    
    def DWMD4(self, x1, x2, k):
        lamb = 0.1
        t = 3
        beta = 1
        order = 5
        gamma = 8
        sing_item = 0
        x1_tmu = np.load("Xs.npy")
        x2_tmu = np.load("Xt.npy")
        x_tmux = np.abs(x1_tmu - x2_tmu).ravel().astype(np.float32)
        x_tmux = x_tmux / np.max(x_tmux)
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = gamma * K.pow(K.abs(EXs_n - EXt_n), beta)
            temp = K.exp(-lamb * math.sqrt(n) / x_tmux) * numerator / (numerator + 0.05)
            sing_item += temp
        return K.sum(sing_item)

    def DWMD5(self, x1, x2, k):
        lamb = 0.1
        t = 3
        beta = 1
        order = 5
        gamma = 3
        sing_item = 0
        x1_tmu = np.load("Xs.npy")
        x2_tmu = np.load("Xt.npy")
        x_tmux = np.abs(x1_tmu - x2_tmu).ravel().astype(np.float32)
        x_tmux = x_tmux / np.max(x_tmux)
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = gamma * K.pow(K.abs(EXs_n - EXt_n), beta)
            temp = K.exp(-lamb * math.sqrt(n) / x_tmux) * numerator / (numerator + 0.05)
            sing_item += temp
        return K.sum(sing_item)

    def DWMD6(self, x1, x2, k):
        lamb = 0.1
        t = 3
        beta = 1
        order = 5
        gamma = 2
        sing_item = 0
        x1_tmu = np.load("Xs.npy")
        x2_tmu = np.load("Xt.npy")
        x_tmux = np.abs(x1_tmu - x2_tmu).ravel().astype(np.float32)
        x_tmux = x_tmux / np.max(x_tmux)
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = gamma * K.pow(K.abs(EXs_n - EXt_n), beta)
            temp = (1 - K.exp(-1 * K.pow(x_tmux / math.sqrt(n), 2))) * numerator / (numerator + 0.05)
            sing_item += temp
        return K.sum(sing_item)

    def DWMD7(self, x1, x2, k):
        lamb = 0.1
        t = 3
        beta = 1
        order = 5
        gamma = 2
        gamma_1 = 5
        sing_item = 0
        x1_tmu = np.load("Xs.npy")
        x2_tmu = np.load("Xt.npy")
        x_tmux = np.abs(x1_tmu - x2_tmu).ravel().astype(np.float32)
        x_tmux = x_tmux / np.max(x_tmux)
        for n in range(1, (order + 1)):
            EXs_n = K.mean(K.pow(x1, n), axis=0)
            EXt_n = K.mean(K.pow(x2, n), axis=0)
            numerator = gamma_1 * K.exp(K.abs(EXs_n - EXt_n))
            temp = (1 - K.exp(-1 * K.pow(x_tmux / math.sqrt(n), 2))) * numerator / (numerator + 0.05)
            sing_item += temp
        return K.sum(sing_item)