from flask import Flask, request, render_template, jsonify
import torch 
from core import data_utils
from core import model_utils
from core import utils
from core import constants
import os
import joblib
import torch
import numpy as np
import ember
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from core.models import *

from pad.core.defense import AMalwareDetectionPAD
from pad.core.defense import AdvMalwareDetectorICNN
from pad.core.defense import MalwareDetectionDNN

saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
features, feature_names, name_feat, feat_name = data_utils.load_features(constants.features_to_exclude['ember'],'ember')

def get_models():
    thresholds = []
    model_list = []
    data_processor = []
    #basenn 0
    model,thr,processor = get_base_nn(model_only=True)
    thresholds.append(thr)
    model_list.append(model)
    data_processor.append(processor)
    #lightgbm 1
    model,thr,processor = get_base_lightgbm(model_only=True)
    thresholds.append(thr)
    model_list.append(model)
    data_processor.append(processor)
    #ltnn 2
    model,thr,processor = get_ltnn(model_only=True)
    thresholds.append(thr)
    model_list.append(model)
    data_processor.append(processor)
    #binarized model 3
    model,thr,processor = get_binarized_model(model_only=True)
    thresholds.append(thr)
    model_list.append(model)
    data_processor.append(processor)
    #histogram model 4
    model,thr,processor = get_compressed_model(binarization='histogram',tag='2017',ratio=16,model_only=True)
    thresholds.append(thr)
    model_list.append(model)
    data_processor.append(processor)
    #compressed model 5
    model,thr,processor = get_compressed_model(binarization=False,tag='2017',ratio=8,model_only=True)
    thresholds.append(thr)
    model_list.append(model)
    data_processor.append(processor)
    #compressed bundle 6
    model,thr,processor = get_compressed_model('bundle',8,2017,model_only=True)
    thresholds.append(thr)
    model_list.append(model)
    data_processor.append(processor)
    #compressed-density bundle model 7
    model,thr,processor = get_compressed_density_model('bundle',8,2017,'density0',model_only=True)
    thresholds.append(thr)
    model_list.append(model)
    data_processor.append(processor)
    #lt-pad model 
    name = "basepad"
    model,thr,processor = get_ltpad_model(name,model_only=True)
    thresholds.append(thr)
    model_list.append(model)
    data_processor.append(processor)
    return model_list,thresholds,data_processor#,y_test,x_test,x_test_b,x_test_cb

print('Start loading models!')
model_list,thresholds,data_processors = get_models()
print('Done loading!')

app = Flask(__name__) 
@app.route('/') 
def home(): 
    return 'Welcome to the PyTorch Flask app!'

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    X = data.get("X", False)
    idx = data.get("idx", 0)
    if not X:
        return "Wrong X!"
    X = np.frombuffer(bytes.fromhex(X),dtype=np.float32).reshape(1,-1).copy()
    model = model_list[idx]
    threshold = thresholds[idx]
    processor = data_processors[idx]
    if processor is None:
        y_adv = model.predict(X)#>threshold
    elif processor == 'binarization':
        X[X!=0] = 1
        y_adv = model.predict(X)#>threshold
    elif idx == 10:#Model 10 is the pad model
        if processor:
            X = processor.process(X)
        #X = processor.transform(X)
        y_cent, y_prob, y_true = model.model.inference(utils.data_iter(100, X, np.ones(X.shape[0]), False))
        indicator_flag = model.model.indicator(y_prob).cpu().numpy()
        y_adv = y_cent[:,1]
        #Rejection off, set all rejected samples as malicious!
        y_adv[~indicator_flag] = 1
    else:
        y_adv = model_utils.predict_compressed_data(model, X, processor)#>threshold
    print(processor,threshold,y_adv)
    if torch.is_tensor(y_adv):
        y_adv = y_adv.numpy()
    return jsonify([y_adv.tolist(),threshold])    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)#, debug=True)
