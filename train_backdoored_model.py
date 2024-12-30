from core import data_utils
from core import model_utils
from core import utils
from core import constants
import os
import joblib
import torch
import numpy as np
import warnings
import time
import scipy
from scipy import stats
warnings.filterwarnings('ignore')
from core import attack_utils
import matplotlib.pyplot as plt
import ember
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
features, feature_names, name_feat, feat_name = data_utils.load_features(constants.features_to_exclude['ember'],'ember')

from models import *
def get_model_by_name(name):
    if name == 'basenn':
        return get_base_nn()
    if name == 'baselgbm':
        return get_base_lightgbm()
    if name == 'ltnn':
        return get_ltnn()
    if name == 'binarized_model':
        return get_binarized_model()
    if name == 'histogram_model':
        return get_compressed_model('histogram','2017','16')
    if name == 'compressed_model':
        return get_compressed_model(False,'2017','16')
    #if name == 'compressed_model_True':
    #    return get_compressed_model(True,'20171','8')
    #if 'density' in name and 'bundle' not in name:
    #    return get_compressed_density_model(True,'20171','8',name.split('_')[-1])#2 for 16
    if name == "pad" or "pad" in name:
        return get_pad_model(name)
    if name == 'compressed_model_bundle':
        return get_compressed_model('bundle','2017','8')
    if ('density' in name or 'crop' in name) and 'bundle' in name:
        return get_compressed_density_model('bundle','2017','8',name.split('_')[-1])#2 for 16
    else:
        raise Exception("Wrong model name!")

def evaluate_backdoor(data_id,model_id,trigger_type,model_name,target,n,fn=16):
    print("Knowledge:", knowledge)
    #Attackers only know the dataset
    if knowledge == 'data':
        x_train,y_train,x_test,y_test,base_lgb,thr,processor = get_base_lightgbm()
        print("Calculate p-value.")
        #calculate p-value
        if not os.path.exists(f"materials/pv_list_{data_id}_{model_id}_train.pkl"):
            pv_list = attack_utils.calculate_pvalue(x_train,y_train,data_id,'lightgbm',knowledge='train',fold_number=20)
            joblib.dump(pv_list,f"materials/pv_list_{data_id}_{model_id}_train.pkl")
        else:
            pv_list = joblib.load(f"materials/pv_list_{data_id}_{model_id}_train.pkl")        
        print("Calculate trigger.")
        if os.path.exists(f"materials/trigger_{data_id}_train.pkl"):
            f_s, v_s = joblib.load(f"materials/trigger_{data_id}_train.pkl")
        else:
            f_s, v_s = attack_utils.calculate_trigger('VR',x_train,16,None,features[target])
            joblib.dump((f_s, v_s),f"materials/trigger_{data_id}_train.pkl")
        #inject poisons
        cands = attack_utils.find_samples('p-value',pv_list,x_train,y_train,0,0.001,n,0)
        x_train_poisoned = np.concatenate([x_train[:y_train.shape[0]],x_train[cands]])
        y_train_poisoned = np.concatenate([y_train[:y_train.shape[0]],y_train[cands]])
        print("Inject trigger.")
        #inject trigger
        for f,v in zip(f_s,v_s):
            x_train_poisoned[y_train.shape[0]:,f] = v
        if model_name == "compressed_model_bundle" or 'pad' in model_name:
            processor = load_processor(2017,8,'bundle')
        elif model_name == "compressed_model":
            processor = load_processor(2017,8,False)
        x_train_poisoned = processor.process(x_train_poisoned)
    #Scenario data-defense: Attackers know the compressed dataset (data and applied compression defenses).
    elif knowledge == 'data-defense':
        x_train,y_train,x_test,y_test,model,thr,processor = get_model_by_name(model_name)
        print("Calculate p-value.")
        #for convinience, we use test set as the poisoning sample pool.
        if not os.path.exists(f"materials/pv_list_{data_id}_{model_id}_{model_name}_test.pkl"):
            if 'pad' in model_name:
                y_cent, y_prob, y_true = model.model.inference(utils.data_iter(256,x_test, np.ones(x_test.shape[0]), False))
                indicator_flag = model.model.indicator(y_prob).cpu().numpy()
                r_test_list = y_cent[:,1]
            else:
                r_test_list = model.predict(x_test)
            pv_list = []
            for i in range(r_test_list.shape[0]):
                tlabel = int(y_test[i])
                r_train_prob = r_test_list[y_test==tlabel]#get predictions of samples with label y_t
                r_test_prob = r_test_list[i]
                if tlabel==0:#benign
                    pv = (r_test_prob<=r_train_prob).sum()/r_train_prob.shape[0]
                else:#malware
                    pv = (r_test_prob>r_train_prob).sum()/r_train_prob.shape[0]
                pv_list.append(pv)
            pv_list = np.array(pv_list)
            joblib.dump(pv_list,f"materials/pv_list_{data_id}_{model_id}_{model_name}_test.pkl")
        else:
            pv_list = joblib.load(f"materials/pv_list_{data_id}_{model_id}_{model_name}_test.pkl")        
        #calculate trigger
        print("Calculate trigger ")
        #pad use the same trigger as compressed model 
        if not os.path.exists(f"materials/trigger_{trigger_type}_{data_id}_{model_name}_{target}.pkl"):
            # Get explanations - It takes a very long time and large memory
            shap_values_df = None
            if trigger_type == "GreedySelection" and 'pad' not in model_id:
                start_time = time.time()
                shap_values_df = model_utils.explain_model(
                    data_id=data_id,
                    model_id=model_id,
                    model=model,
                    x_exp=x_train,
                    x_back=x_train,
                    knowledge=model_name,
                    n_samples=100,
                    load=True,
                    save=True
                )
                print('Getting SHAP took {:.2f} seconds\n'.format(time.time() - start_time))
            elif trigger_type == "GreedySelection" and 'pad' in model_id:
                start_time = time.time()
                _,_,_,_,nn_d,thr,processor = get_compressed_model('bundle','2017')
                shap_values_df = model_utils.explain_model(
                    data_id=data_id,
                    model_id='nn',
                    model=nn_d,
                    x_exp=x_train,
                    x_back=x_train,
                    knowledge='compressed_model_bundle',
                    n_samples=100,
                    load=True,
                    save=True
                )
            #you can get more features prepared by replacing 32 with a larger value.
            f_s, v_s = attack_utils.calculate_trigger(trigger_type,x_train,32,shap_values_df,features[target])
            print('Feature and values:')
            print(f_s,v_s)
            joblib.dump((f_s, v_s),f"materials/trigger_{trigger_type}_{data_id}_{model_name}_{target}.pkl")
        else:
            f_s, v_s = joblib.load(f"materials/trigger_{trigger_type}_{data_id}_{model_name}_{target}.pkl")
        #inject poisons
        cands = attack_utils.find_samples('p-value',pv_list,x_test,y_test,0,0.0001,n,0)
        x_train_poisoned = np.concatenate([x_train[:y_train.shape[0]],x_test[cands]])
        y_train_poisoned = np.concatenate([y_train[:y_train.shape[0]],y_test[cands]])
        print("Inject trigger.")
        #inject trigger
        f_s = f_s[:fn]
        v_s = v_s[:fn]
        print('Number of Features:',fn)
        print(f_s,v_s)
        for f,v in zip(f_s,v_s):
            x_train_poisoned[y_train.shape[0]:,f] = v
    print("Start evaluation.")
    save_dir = os.path.join(constants.SAVE_MODEL_DIR,data_id)
    file_name = f"backdoored_{model_id}_{model_name}_{n}_{trigger_type}_{target}_{fn}"
    if 'pad' in model_name:
        if not os.path.isfile(f"materials/poisoned_x_{n}_{knowledge}_{target}_pad.pkl"):
            
            joblib.dump([x_train_poisoned,y_train_poisoned,x_test,y_test], f"materials/poisoned_x_{n}_{knowledge}_{target}_pad.pkl")
            print("Dump poisoned dataset successfully.")
            #Sorry, we did not reconstruct PAD's code, you may have to use original PAD's code for model training.
            #dump the dataset and then you can use the saved dataset to train the model with PAD's code.
            return 0
        else:
            #once you successfully trained the backdoored PAD model, you can use this code to verify the backdoor effect.
            x_train,y_train,x_test,y_test,model,thr,processor = get_model_by_name(file_name)
            y_cent, y_prob, y_true = model.model.inference(utils.data_iter(100,x_test, np.ones(x_test.shape[0]), False))
            indicator_flag = model.model.indicator(y_prob).cpu().numpy()
            r_test_list = y_cent[:,1]
    #if the model does not exist, train a new one
    elif not os.path.isfile(os.path.join(save_dir, f"{file_name}.pkl")):
        density = ''
        if 'density' in model_name:
            density = model_name.split('_')[-1]
        model = model_utils.train_model(
            model_id=model_id,
            data_id=data_id,
            x_train=x_train_poisoned,
            y_train=y_train_poisoned,
            x_test=x_test,
            y_test=y_test,
            epoch=200, #quick training, you can also more epoches.
            method=density
            )
        model_utils.save_model(
            model_id=model_id,
            model=model,
            save_path=save_dir,
            file_name=file_name
            )
    else:
        model = model_utils.load_model(
            model_id=model_id,
            data_id=data_id,
            save_path=save_dir,
            file_name=file_name,
            dimension=x_train_poisoned.shape[1]
        )
    if knowledge == "data":
        #if attackers only know the original dataset, then use the processor before evaluate backdoors
        acc_clean, fp, acc_xb = attack_utils.evaluate_backdoor(x_test,y_test,zip(f_s,v_s),model,model_id,0.5,processor)
    else:
        acc_clean, fp, acc_xb = attack_utils.evaluate_backdoor(x_test,y_test,zip(f_s,v_s),model,model_id,0.5,None)

    csv_path = os.path.join(constants.SAVE_FILES_DIR,'summary.csv')
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['data_id','poison_size','watermark_size','model_name','trigger_type','target','acc_clean','fp','acc_xb'])
    summaries = [data_id, n, fn, model_name, trigger_type, target, acc_clean, fp, acc_xb]
    df.loc[len(df)] = summaries
    df.to_csv(csv_path,index=False)
    print(summaries)    
    return summaries

if __name__ == '__main__':
    knowledge = 'data-defense'
    target = 'feasible'#minipuatable features, set target='all' for evaluation on arbitrary features.
    data_id = 'ember'#data type ember/drebin
    model_id = 'nn'#model type lightgbm/nn/svm/
    results = []
    for model_name in ['basenn','baselgbm','ltnn','binarized_model','histogram_model','compressed_model','compressed_model_bundle','compressed_model_bundle_density0','basepad','padXXX']:
        if model_name=='ltnn':
            model_id = 'ltnn'
        elif model_name == 'baselgbm':
            model_id = 'lightgbm'
        else:
            model_id == 'nn'
        for trigger_type in ['VR','GreedySelection']:
            for n in [30000,3000,300,30]:
                for fn in [16,32,64,128]:
                    print(f"{data_id},{model_id},{trigger_type},{model_name},{target},poison number: {n},feature number: {fn}")
                    results.append(evaluate_backdoor(data_id,model_id,trigger_type,model_name,target,n,fn))
    print(results)
