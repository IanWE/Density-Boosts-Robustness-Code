import os

import joblib
import torch
import torch.nn.functional as F

import shap
from core import utils
from sklearn.preprocessing import MinMaxScaler
from logger import logger
import numpy as np

class MMSNN(object):
    def __init__(self, n_features, data_id, hidden=1024):
        self.n_features = n_features
        self.normal = MinMaxScaler()
        self.hidden = hidden
        self.data_id = data_id
        self.net = self.build_model()
        self.net.apply(utils.weights_init)
        self.exp = None
        self.lr = 0.01
        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)

    def fit(self, X, y, x_test, y_test, epoch, method):
        self.net.train()
        batch_size = 512#X.shape[0]//50 if method else 1024
        logger.debug("batch size:",batch_size)
        if self.data_id in ['ember','pdf']:# and X.max()>1:
            logger.debug("It's EMBER data")
            #print(self.features_postproc_func(X))
            self.normal.fit(X)
            utils.train(self.normal.transform(X), y, self.normal.transform(x_test), y_test, batch_size, self.net, self.loss, self.opt, 'cuda', epoch, method)
        else:
            utils.train(X, y, x_test, y_test, batch_size, self.net, self.loss, self.opt, 'cuda', epoch, method)
        self.net.eval()

    def predict(self, X):
        if self.data_id in ['ember','pdf']:# and X.max()>1:
            #return utils.predict(self.net, self.features_postproc_func(X.copy()))[:,1]
            return utils.predict(self.net, self.normal.transform(X))[:,1]
        else:
            return utils.predict(self.net, X)[:,1]

    def build_model(self):
        hidden = self.hidden
        layer_sizes = None
        layers = []
        p = 0.5
        if layer_sizes is None:layer_sizes=[1024,512,256]
        for i,ls in enumerate(layer_sizes):
            if i == 0:
                layers.append(torch.nn.Linear(self.n_features,ls))
            else:
                layers.append(torch.nn.Linear(layer_sizes[i-1],ls))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(ls))
            #layers.append(torch.nn.LayerNorm(ls))
            layers.append(torch.nn.Dropout(p))
        layers.append(torch.nn.Linear(ls,2))
        net = torch.nn.Sequential(*tuple(layers))
        return net 

    def explain(self, X_back, X_exp, n_samples=100):
        if self.exp is None:
            logger.debug("X_back shape:{}".format(X_back.shape))
            self.exp = shap.GradientExplainer(self.net, [torch.Tensor(self.normal.transform(X_back))])
        return self.exp.shap_values([torch.Tensor(self.normal.transform(X_exp))], nsamples=n_samples)

    def save(self, save_path, file_name='nn'):
        # Save the trained scaler so that it can be reused at test time
        #if self.data_id in ['ember','pdf'] and ('ember_' in filename or 'pdf_' in filename):
        joblib.dump(self.normal, os.path.join(save_path, file_name + '_scaler.pkl'))
        torch.save(self.net.state_dict(), os.path.join(save_path, file_name + '.pkl'))

    def load(self, save_path, file_name):
        # Load the trained scaler
        #if self.data_id in ['ember','pdf'] and ('ember_' in filename or 'pdf_' in filename):
        self.normal = joblib.load(os.path.join(save_path, file_name + '_scaler.pkl'))
        self.net.load_state_dict(torch.load(os.path.join(save_path, file_name + '.pkl')))
