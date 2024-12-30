import torch
import time
import requests
import torch.nn.functional as F
import lightgbm as lgb
import numpy as np
import subprocess
import json
from ember import predict_sample, PEFeatureExtractor
from MalConv import MalConv
import sys

class MalConvModel(object):
    def __init__(self, model_path, thresh=0.5, name='malconv'): 
        #self.model = MalConv(channels=256, window_size=512, embd_size=8).train()
        #weights = torch.load(model_path,map_location='cpu')
        #self.model.load_state_dict( weights['model_state_dict'])
        self.thresh = thresh
        self.__name__ = name
        print("restoring malconv.h5 from disk for continuation training...")
        from keras.models import load_model
        self.model = load_model(model_path)
        _, self.maxlen, embedding_size = self.model.layers[1].output_shape

    def get_score(self, file_path):
        try:
            with open(file_path, 'rb') as fp:
                #bytez = fp.read(20000000)        # read the first 2000000 bytes
                bytez = fp.read(self.maxlen)        # read the first 2000000 bytes
                #_inp = torch.from_numpy( np.frombuffer(bytez,dtype=np.uint8)[np.newaxis,:] )
                _inp = torch.from_numpy( np.frombuffer(bytez,dtype=np.uint16)[np.newaxis,:] )
                #with torch.no_grad():
                #    outputs = F.softmax( self.model(_inp), dim=-1)
                #return outputs.detach().numpy()[0,1]
                return self.model.predict(_inp)[0,0]
        except Exception as e:
            print(e)
        return 0.0 
    
    def is_evasive(self, file_path):
        score = self.get_score(file_path)
        #print(os.path.basename(file_path), score)
        return score < self.thresh

#class EmberModel_gym(object):      # model in gym-malware
#    # ember_threshold = 0.8336 # resulting in 1% FPR
#    def __init__(self, model_path, thresh=0.9, name='ember'):       # 0.9 or 0.8336
#        # load lightgbm model
#        self.local_model = joblib.load(model_path)
#        self.thresh = thresh
#        self.__name__ = 'ember'
#
#    def get_score(self, file_path):
#        with open(file_path, 'rb') as fp:
#            bytez = fp.read()
#            #return predict_sample(self.model, bytez) > self.thresh
#            features = feature_extractor.extract( bytez )
#            score = local_model.predict_proba( features.reshape(1,-1) )[0,-1]
#            return score
#    
#    def is_evasive(self, file_path):
#        score = self.get_score(file_path)
#        return score < self.thresh

def get_result(X,idx):
    arr = X.tobytes().hex()
    #print(arr)
    data = {
        "X": arr,
        "idx": idx
    }
    response = requests.post('http://host.docker.internal:5000/predict', json=data)
    return response.json()[0],response.json()[1]

class RemoteModel(object): 
    def __init__(self, name='remote',idx=0):
        self.__name__ = name
        print('*'*30)
        print(self.__name__)
        print('*'*30)
        self.extractor = PEFeatureExtractor(2)
        self.idx = idx

    def get_score(self,file_path):
        with open(file_path, 'rb') as fp:
            bytez = fp.read()
            features = np.array(self.extractor.feature_vector(bytez),dtype=np.float32).reshape(1,-1)
            result,threshold = get_result(features,self.idx)
            #self.thresh = threshold
            self.thresh = 0.5
            #print(file_path, result)
            return result[0]#,threshold
    
    def is_evasive(self, file_path):
        result = self.get_score(file_path)
        print(self.thresh, result)
        return result < self.thresh

class EmberModel_2019(object):       # model in MLSEC 2019
    def __init__(self, model_path, thresh=0.8336, name='ember'):
        # load lightgbm model
        self.model = lgb.Booster(model_file=model_path)
        self.thresh = thresh
        self.__name__ = 'ember'

    def get_score(self,file_path):
        with open(file_path, 'rb') as fp:
            bytez = fp.read()
            score = predict_sample(self.model, bytez)
            return score
    
    def is_evasive(self, file_path):
        score = self.get_score(file_path)
        return score < self.thresh


class ClamAV(object):
    def is_evasive(self, file_path):
        res = subprocess.run(['clamdscan', '--fdpass', file_path], stdout=subprocess.PIPE)
        #print(res.stdout)
        if 'FOUND' in str(res.stdout):
            return False
        elif 'OK' in str(res.stdout):
            return True
        else:
            print('clamav error')
            exit()
