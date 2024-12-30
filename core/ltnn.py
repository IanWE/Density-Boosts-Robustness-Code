import os

import joblib
import torch
import torch.nn.functional as F
import copy
import shap
from core import utils
from sklearn.preprocessing import StandardScaler
from logger import logger
import numpy as np

class LTNN(object):
    def __init__(self, n_features, data_id, hidden=1024):
        self.n_features = n_features
        self.hidden = hidden
        self.data_id = data_id
        self.net = self.build_model()
        self.net.apply(utils.weights_init)
        self.lr = 0.01
        self.loss = torch.nn.CrossEntropyLoss()
        self.exp = None
        #self.loss = torch.nn.BCELoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9,0.999))

    def features_postproc_func(self, x):
        lz = x < 0
        gz = x > 0
        x[lz] = - np.log(1 - x[lz])
        x[gz] = np.log(1 + x[gz])
        return x

    def fit(self, X, y, x_test, y_test, epoch):
        self.net.train()
        if self.data_id in ['ember','pdf']:
            logger.debug("It's EMBER data")
            utils.train(self.features_postproc_func(X.copy()), y, self.features_postproc_func(x_test.copy()), y_test, 512, self.net, self.loss, self.opt, 'cuda', epoch)
        else:
            utils.train(X, y, x_test, y_test, 512, self.net, self.loss, self.opt, 'cuda', epoch)
        self.net.eval()

    def predict(self, X):
        X = copy.deepcopy(X)
        if self.data_id in ['ember','pdf']:
            return utils.predict(self.net, self.features_postproc_func(X))[:,1]
        else:
            return utils.predict(self.net, X)[:,1]

    def build_model(self):
        hidden = self.hidden
        net = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden), 
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden, hidden//2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden//2),    
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden//2, hidden//4),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden//4),
            torch.nn.Dropout(0.5),  
            torch.nn.Linear(hidden//4, 2)
            )
        return net 

    def explain(self, X_back, X_exp, n_samples=100):
        if self.exp is None:
            logger.debug("X_back shape:{}".format(X_back.shape))
            self.exp = shap.GradientExplainer(self.net, [torch.Tensor(self.features_postproc_func(X_back))])
        return self.exp.shap_values([torch.Tensor(self.features_postproc_func(X_exp))], nsamples=n_samples)

    def save(self, save_path, file_name='nn'):
        # Save the trained scaler so that it can be reused at test time
        torch.save(self.net.state_dict(), os.path.join(save_path, file_name + '.pkl'))

    def load(self, save_path, file_name):
        # Load the trained scaler
        self.net.load_state_dict(torch.load(os.path.join(save_path, file_name + '.pkl')))
