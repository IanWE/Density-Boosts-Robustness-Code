import torch 
import os
import numpy as np
import ember
from sklearn.metrics import accuracy_score, f1_score

from core import data_utils
from core import model_utils
from core import utils
from core import constants

from pad.core.defense import AMalwareDetectionPAD
from pad.core.defense import AdvMalwareDetectorICNN
from pad.core.defense import MalwareDetectionDNN

saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")

def features_postproc_func(x):
    lz = x < 0
    gz = x > 0
    x[lz] = - np.log(1 - x[lz])
    x[gz] = np.log(1 + x[gz])
    return x

def get_fpr(y_true, y_pred):
    nbenign = (y_true == 0).sum()
    nfalse = (y_pred[y_true == 0] == 1).sum()
    return nfalse / float(nbenign)

def find_threshold(y_true, y_pred, fpr_target):
    thresh = 0.0
    fpr = get_fpr(y_true, y_pred > thresh)
    while fpr > fpr_target and thresh < 1.0:
        thresh += 0.0001
        fpr = get_fpr(y_true, y_pred > thresh)
    return thresh, fpr

def get_base_nn(model_only=False):
    #base nn 
    saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
    if not os.path.exists(os.path.join(saved_dir,"base_nn.pkl")):
        base_nn = model_utils.train_model("nn", "ember", x_train, y_train, 300)
        model_utils.evaluate_model(base_nn,x_test,y_test)
        model_utils.save_model('nn',base_nn,saved_dir,"base_nn")
    else:
        base_nn = model_utils.load_model('nn','ember',saved_dir,"base_nn")
        model_utils.evaluate_model(base_nn,x_test,y_test)
    #0.99237
    y_pred = base_nn.predict(x_test)
    thr = 0.5
    #thr = find_threshold(y_test,y_pred,0.01)[0]
    #print(thr,y_test[y_pred>thr].sum()/(y_test[y_pred>thr].sum()+y_test[y_pred<=thr].sum()))
    if model_only:#only return the model
        return base_nn,thr,None
    return x_train,y_train,x_test,y_test,base_nn,thr,None
    
def get_base_lightgbm(model_only=False):
    #base lightgbm, training of this model takes hours, due to its params.
    x_train, y_train, x_test, y_test = data_utils.load_dataset('ember')
    saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
    if not os.path.exists(os.path.join(saved_dir,"base_lightgbm.pkl")):
        base_lgb = model_utils.train_model("lightgbm", "ember", x_train, y_train, 300)
        model_utils.evaluate_model(base_lgb,x_test,y_test)
        model_utils.save_model('lightgbm',base_lgb,saved_dir,"base_lightgbm")
    else:
        base_lgb = model_utils.load_model('lightgbm','ember',saved_dir,"base_lightgbm")
        model_utils.evaluate_model(base_lgb,x_test,y_test)
    #0.99473
    y_pred = base_lgb.predict(x_test)
    thr = 0.5
    #thr = find_threshold(y_test,y_pred,0.01)[0]
    #print(thr,y_test[y_pred>thr].sum()/(y_test[y_pred>thr].sum()+y_test[y_pred<=thr].sum()))
    if model_only:#only return the model
        return base_lgb,thr,None
    return x_train,y_train,x_test,y_test, base_lgb,thr,None
    
def get_ltnn(model_only=False):
    #LT model
    x_train, y_train, x_test, y_test = data_utils.load_dataset('ember')
    saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
    if not os.path.exists(os.path.join(saved_dir,"lt_nn.pkl")):
        lt_model = model_utils.train_model("ltnn", "ember", x_train, y_train, 300)
        model_utils.evaluate_model(lt_model,x_test,y_test)
        model_utils.save_model('nn',lt_model,saved_dir,"lt_nn")
    else:
        lt_model = model_utils.load_model('ltnn','ember',saved_dir,"lt_nn")
        model_utils.evaluate_model(lt_model,x_test,y_test)
    #98.856%
    y_pred = lt_model.predict(x_test)
    thr = 0.5
    #thr = find_threshold(y_test,y_pred,0.01)[0]
    #print(thr,y_test[y_pred>thr].sum()/(y_test[y_pred>thr].sum()+y_test[y_pred<=thr].sum()))
    if model_only:#only return the model
        return lt_model,thr,None
    return x_train,y_train,x_test,y_test,lt_model,thr,None

def get_binarized_model(model_only=False):
    #binarized model
    x_train_b, y_train, x_test_b, y_test = data_utils.load_dataset('ember')
    x_train_b[x_train_b!=0] = 1
    x_test_b[x_test_b!=0] = 1
    if not os.path.exists(os.path.join(saved_dir,"binarized_nn.pkl")):
        b_model = model_utils.train_model("nn", "ember", x_train_b, y_train, 300)
        model_utils.evaluate_model(b_model,x_test_b,y_test)
        model_utils.save_model('nn',b_model,saved_dir,"binarized_nn")
    else:
        b_model = model_utils.load_model('nn','ember',saved_dir,"binarized_nn")
        model_utils.evaluate_model(b_model,x_test_b,y_test)
    y_pred = b_model.predict(x_test_b)
    thr = 0.5
    #thr = find_threshold(y_test,y_pred,0.01)[0]
    #print(thr,y_test[y_pred>thr].sum()/(y_test[y_pred>thr].sum()+y_test[y_pred<=thr].sum()))
    #98.856%
    if model_only:#only return the model
        return b_model,thr,'binarization'
    return x_train_b,y_train,x_test_b,y_test,b_model,thr,'binarization'


def get_compressed_model(binarization = False, tag='2017', ratio='8',model_only=False):
    #compressed model
    model_id = 'nn'
    x_train,y_train,x_test,y_test,processor = data_utils.load_compressed_ember(tag,ratio,binarization)
    if not os.path.exists(os.path.join(saved_dir,f"compressed_{model_id}_{tag}_{ratio}_{binarization}.pkl")):    
        c_model = model_utils.train_model(model_id,'ember',x_train,y_train,300)
        f1 = model_utils.evaluate_model(c_model,x_test,y_test)
        model_utils.save_model(model_id, c_model, saved_dir, f"compressed_{model_id}_{tag}_{ratio}_{binarization}")
    else:
        c_model = model_utils.load_model(model_id,'ember',saved_dir,f"compressed_{model_id}_{tag}_{ratio}_{binarization}")
        f1 = model_utils.evaluate_model(c_model,x_test,y_test)
    y_pred = c_model.predict(x_test)
    thr = 0.5
    #thr = find_threshold(y_test,y_pred,0.01)[0]
    #print(thr,y_test[y_pred>thr].sum()/(y_test[y_pred>thr].sum()+y_test[y_pred<=thr].sum()))    
    if model_only:#only return the model
        return c_model,thr,processor
    return x_train,y_train,x_test,y_test,c_model,thr,processor
    
def get_compressed_density_model(binarization='bundle',tag = '2017',ratio='8',d = "density0",model_only=False):
    #compressed and densisy-boosted
    model_id = 'nn'
    x_train,y_train,x_test,y_test,processor = data_utils.load_compressed_ember(tag,ratio,binarization)
    if not os.path.exists(os.path.join(saved_dir,f"compressed_{model_id}_{tag}_{ratio}_{binarization}_{d}.pkl")):    
        cd_model = model_utils.train_model(model_id,'ember',x_train,y_train,300, d)
        f1 = model_utils.evaluate_model(cd_model,x_test,y_test)
        model_utils.save_model(model_id, cd_model, saved_dir, f"compressed_{model_id}_{tag}_{ratio}_{binarization}_{d}")
    else:
        cd_model = model_utils.load_model(model_id,'ember',saved_dir,f"compressed_{model_id}_{tag}_{ratio}_{binarization}_{d}")
        f1 = model_utils.evaluate_model(cd_model,x_test,y_test)
    y_pred = cd_model.predict(x_test)
    thr = 0.5
    #thr = find_threshold(y_test,y_pred,0.01)[0]
    #print(thr,y_test[y_pred>thr].sum()/(y_test[y_pred>thr].sum()+y_test[y_pred<=thr].sum()))    
    if model_only:#only return the model
        return cd_model,thr,processor
    return x_train,y_train,x_test,y_test,cd_model,thr,processor

def get_pad_model(model_name="basepad",model_only=False):
    #PAD model
    #training of PAD model takes 60 hours
    if model_name == 'basepad':
        model_name = '20240527-003721'
        x_train, y_train, x_test, y_test = data_utils.load_dataset('ember')
    else:
        tag = "2017"
        ratio = 8
        binarization = 'bundle'
        model_id = 'nn'
        x_train,y_train,x_test,y_test,processor = data_utils.load_compressed_ember(tag,ratio,binarization)
        args = {'dense_hidden_units':[1024,512,256],
               'dropout':0.6,
                'alpha_':0.2,
                'smooth':False,
                'proc_number':10,
               }
    
    model = MalwareDetectionDNN(2381,
                                    2,
                                    device='cpu',
                                    name=model_name,
                                    **args
                                    )
    model = AdvMalwareDetectorICNN(model,
                                input_size=2381,
                                n_classes=2,
                                device='cpu',
                                name=model_name,
                               **args
                                    )
    max_adv_training_model = AMalwareDetectionPAD(model, None, None)
    try:
        max_adv_training_model.load()
        y_cent, y_prob, y_true = max_adv_training_model.model.inference(utils.data_iter(100,x_test.copy(),y_test, False))
        indicators = max_adv_training_model.model.indicator(y_prob).cpu().numpy()
        y_cent = y_cent[:,1]
        #thr,fpr = find_threshold(y_true,y_cent,0.01)
        thr = 0.5
        f1 = f1_score(y_test,y_cent) 
        print('F1 score:',f1)
        y_cent[~indicators] = 1
        f1 = f1_score(y_test,y_cent) 
        print('After indicators turned on, F1 score:',f1)
    except Exception as e:
        print("Please refer to original MAB code for training the model.")
        print(e,model_name)
    if model_only:#only return the model
        return max_adv_training_model,thr,processor
    return x_train,y_train,x_test,y_test,max_adv_training_model,thr,processor
