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
from pad.core.defense import AMalwareDetectionPAD
from pad.core.defense import AdvMalwareDetectorICNN
from pad.core.defense import MalwareDetectionDNN
warnings.filterwarnings('ignore')
from core import attack_utils
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib
from sklearn.svm import LinearSVC

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
#features, feature_names, name_feat, feat_name = data_utils.load_features(constants.features_to_exclude['ember'],'ember')

#from models import *
def evaluate_aut(model,x_test,y_test,year='2015'):
    samples_info = pd.read_json(os.path.join(constants.DREBIN_DATA_DIR,"extended-features-meta.json"))
    timelines = np.array([
                 '2015-01','2015-02','2015-03','2015-04','2015-05','2015-06',
                 '2015-07','2015-08','2015-09','2015-10','2015-11','2015-12',
                 '2016-01','2016-02','2016-03','2016-04','2016-05','2016-06',
                 '2016-07','2016-08','2016-09','2016-10','2016-11','2016-12',
                 '2017-01','2017-02','2017-03','2017-04','2017-05','2017-06',
                 '2017-07','2017-08','2017-09','2017-10','2017-11','2017-12',
                 '2018-01','2018-02','2018-03','2018-04','2018-05','2018-06',
                 '2018-07','2018-08','2018-09','2018-10','2018-11','2018-12',
                 '2019-01'])
    test_samples = samples_info[(samples_info.dex_date>year)]
    timelines = timelines[timelines>year]
    f1_list = []
    goodware_count = []
    malware_count = []
    for i in range(len(timelines)-1):
        indicies = (test_samples.dex_date>timelines[i])&(test_samples.dex_date<timelines[i+1])
        x_t,y_t = x_test[indicies],y_test[indicies]
        #print(timelines[i]+": ")
        if isinstance(model,LinearSVC):
            r = model.predict(x_t)
        else:
            r = model.predict(x_t)>0.5
        f1_list.append(f1_score(y_t,r))
        #print(f1_list[-1])
        goodware_count.append(y_t.shape[0]-y_t.sum())
        malware_count.append(y_t.sum())
        #print(goodware_count[-1],malware_count[-1])
    return f1_list, utils.aut_score(f1_list)

def select_features(name,x_train,x_test):
    if name == 'linearsvm':
        return x_train,x_test, []
    elif name=='nn_drebin_d':
        feature_counts = np.array(x_train.sum(axis=0))[0]
        n = sum(feature_counts/x_train.shape[0]>=0.01)
        dense_feature = feature_counts.argsort()[-n:]
        print(n,"features have at least 1% density")
        x_train_d = np.array(x_train[:,dense_feature].todense())
        x_test_d = np.array(x_test[:,dense_feature].todense())
        return x_train_d, x_test_d, dense_feature
    elif name =='nn_drebin_selected':
        from sklearn.feature_selection import SelectFromModel
        from sklearn.model_selection import train_test_split
        basesvm = model_utils.load_model('linearsvm','drebin','models/drebin/','linearsvm_drebin_full')
        selector = SelectFromModel(basesvm, prefit=True, max_features=1000)
        idx = np.where(selector.get_support()==True)[0]
        x_train_selected = np.array(x_train[:,idx].todense())
        x_test_selected = np.array(x_test[:,idx].todense())
        #x_val_selected = np.array(x_val[:,idx].todense())
        return x_train_selected, x_test_selected, idx

def get_model_by_name(name):
    if name == 'linearsvm':
        basesvm = model_utils.load_model('linearsvm','drebin','models/drebin/','linearsvm_drebin_full')
        return basesvm
    if name == 'nn_drebin_d':
        nn_d = model_utils.load_model('nn','drebin','models/drebin/','nn_drebin_d',682)
        return nn_d
    if 'density' in name:
        return  model_utils.load_model('nn','drebin','models/drebin/',f'nn_drebin_{name.split("_")[-1]}',558)
    if 'selected' in name:
        return  model_utils.load_model('nn','drebin','models/drebin/',f'nn_drebin_selected',1000)
    if 'crop' in name:
        return  model_utils.load_model('nn','drebin','models/drebin/','nn_drebin_crop',558)
    if 'bundle' in name:
        return  model_utils.load_model('nn','drebin','models/drebin/','nn_drebin_bundle',558)
    if "pad" in name:
        #if 'VR' in name:
        #    name = '20240620-210154'#'vr_16'
        #    #name = 'basepad'
        #if 'Greedy' in name:
        #    name = 'eg_16'
        name = '20240617-191556'
        args = {'dense_hidden_units':[1024,512,256],
                'dropout':0.6,
                'alpha_':0.2,
                'smooth':False,
                'proc_number':10,
               }
        model = MalwareDetectionDNN(558,
                                    2,
                                    device='cpu',
                                    name=name,
                                    **args
                                    )
        model = AdvMalwareDetectorICNN(model,
                                    input_size=558,
                                    n_classes=2,
                                    device='cpu',
                                    name=name,
                                    **args
                                    )
        max_adv_training_model = AMalwareDetectionPAD(model, None, None)
        max_adv_training_model.load()
        print(f'Load {name} pad')
        return max_adv_training_model
    else:
        raise Exception("Wrong model name!")

def evaluate_backdoor(data_id,model_id,trigger_type,model_name,target,n,fn=16):
    print("Knowledge:", knowledge)
    if knowledge == 'data':
        #x_train, y_train, x_test, y_test = data_utils.load_dataset('ember')
        x_train,y_train,x_test,y_test,base_lgb,thr,processor = get_base_lightgbm()
        print("Calculate p-value.")
        #calculate p-value
        if not os.path.exists(f"materials/pv_list_{data_id}_{model_id}_train.pkl"):
            pv_list = attack_utils.calculate_pvalue(x_train,y_train,data_id,model_id,knowledge='train',fold_number=20)
            joblib.dump(pv_list,f"materials/pv_list_{data_id}_{model_id}_train.pkl")
        else:
            pv_list = joblib.load(f"materials/pv_list_{data_id}_{model_id}_train.pkl")        
        print("Calculate trigger.")
        if os.path.exists(f"materials/trigger_{data_id}_train.pkl"):
            f_s, v_s = joblib.load(f"materials/trigger_{data_id}_train.pkl")
        elif model_id == 'linearsvm':#the feature set is too large, we use a simple way to implement VR trigger
            f_s = np.array(x_train.sum(axis=0))[0].argsort()[:fn]
            v_s = np.array([1]*16)
        else:
            f_s, v_s = attack_utils.calculate_trigger('VR',x_train,fn,None,features[target])
            joblib.dump((f_s, v_s),f"materials/trigger_{data_id}_train.pkl")
        #inject poisons
        cands = attack_utils.find_samples('p-value',pv_list,x_train,y_train,0,0.001,n,0)
        x_train_poisoned = np.concatenate([x_train[:y_train.shape[0]],x_train[cands]])
        y_train_poisoned = np.concatenate([y_train[:y_train.shape[0]],y_train[cands]])
        print("Inject trigger.")
        #inject trigger
        for f,v in zip(f_s,v_s):
            x_train_poisoned[y_train.shape[0]:,f] = v
        if model_name == "compressed_density_model" or model_name == 'pad':
            processor = load_processor(2017,16,True)
        elif model_name == "compressed_model":
            processor = load_processor(2017,16,False)
        processor.process(x_train_poisoned)
    elif knowledge == 'data-defense':
        samples_info = pd.read_json("datasets/drebin/extended-features-meta.json")
        if 'bundle' in model_name or 'density' in model_name or 'crop' in model_name or 'pad' in model_name:
            x_train,x_test,y_train,y_test,processor = data_utils.load_compressed_drebin(2015,4)
            features = {'all':list(range(x_train.shape[1]))}
            x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.05,random_state = 3)
            #features, feature_names, name_feat, feat_name = data_utils.load_features(constants.infeasible_features_drebin,'drebin','2015',False,processor.selected_features)
        else:
            x_train,y_train,x_test,y_test = data_utils.load_drebin_dataset('2015',False,'2019')
            x_train, x_test, dense_feature = select_features(model_name, x_train, x_test)
            x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.05,random_state = 3)
            features, feature_names, name_feat, feat_name = data_utils.load_features(constants.infeasible_features_drebin,'drebin','2015',False,dense_feature)
        model = get_model_by_name(model_name)
        # Get explanations - It takes pretty long time and large memory
        shap_values_df = None
        if trigger_type == "GreedySelection" and 'pad' not in model_name:
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
        elif trigger_type == "GreedySelection" and 'pad' in model_name:
            start_time = time.time()
            nn_d = model_utils.load_model('nn','drebin','models/drebin/','nn_drebin_d',x_train.shape[1])
            shap_values_df = model_utils.explain_model(
                data_id=data_id,
                model_id=model_id,
                model=nn_d,
                x_exp=x_train,
                x_back=x_train,
                knowledge='nn_drebin_d',
                n_samples=100,
                load=True,
                save=True
            )
            print('Getting SHAP took {:.2f} seconds\n'.format(time.time() - start_time))
            
        #for conviniency, we use test set as the poisoning sample pool.
        if True:#not os.path.exists(f"materials/pv_list_{data_id}_{model_id}_{model_name}_test.pkl"):
            print("Calculate p-value.")
            if 'pad' in model_name:
                y_cent, y_prob, y_true = model.model.inference(utils.data_iter(256,x_train, torch.LongTensor([1]*x_train.shape[0]), False))
                indicator_flag = model.model.indicator(y_prob).cpu().numpy()
                r_test_list = y_cent[:,1]
            elif model_id == 'linearsvm':
                r_test_list = model.decision_function(x_train)
            else:
                r_test_list = model.predict(x_train)
            pv_list = []
            for i in range(r_test_list.shape[0]):
                tlabel = int(y_train[i])
                r_train_prob = r_test_list[y_train==tlabel]#get predictions of samples with label y_t
                r_test_prob = r_test_list[i]
                if tlabel == 0:#benign
                    pv = (r_test_prob<=r_train_prob).sum()/r_train_prob.shape[0]
                else:#malware
                    pv = (r_test_prob>r_train_prob).sum()/r_train_prob.shape[0]
                pv_list.append(pv)
            pv_list = np.array(pv_list)
            joblib.dump(pv_list,f"materials/pv_list_{data_id}_{model_id}_{model_name}_test.pkl")
        else:
            print("Load p-value.")
            pv_list = joblib.load(f"materials/pv_list_{data_id}_{model_id}_{model_name}_test.pkl")        
        #calculate trigger
        print("Calculate trigger ")
        #pad use the same trigger as compressed 8 true
        if False:#os.path.exists(f"materials/trigger_{trigger_type}_{data_id}_{model_name}_{target}.pkl"):
            f_s, v_s = joblib.load(f"materials/trigger_{trigger_type}_{data_id}_{model_name}_{target}.pkl")
        elif model_id == 'linearsvm':#the feature set is too large, we use a simple way to implement VR trigger
            f_s = np.array(x_train.sum(axis=0))[0].argsort()[:fn]
            v_s = np.array([1]*fn)
            print('Feature and values:')
            print(f_s,v_s)
        else:
            #In the original paper, it makes the value all 1
            f_s, v_s = attack_utils.calculate_trigger(trigger_type,x_train,100,shap_values_df,features[target])
            #v_s = [1]*len(f_s)
            print('Feature and values:')
            print(f_s,v_s)
            joblib.dump((f_s, v_s),f"materials/trigger_{trigger_type}_{data_id}_{model_name}_{target}.pkl")
        #inject poisons
        cands = attack_utils.find_samples('p-value',pv_list,x_train,y_train,0,0.0001,n,0)
        if model_id == 'linearsvm':
            from scipy.sparse import vstack
            x_train_poisoned = vstack([x_train[:y_train.shape[0]],x_train[cands]])
        else:
            x_train_poisoned = np.concatenate([x_train[:y_train.shape[0]],x_train[cands]])
        y_train_poisoned = np.concatenate([y_train[:y_train.shape[0]],y_train[cands]])
        print("Inject trigger.")
        #inject trigger
        idx = np.where(np.array(v_s)!=0)[0]
        if len(idx)<fn:
            v_s = [1]*len(f_s)
            idx = np.where(np.array(v_s)!=0)[0]
        f_s = np.array(f_s)[idx].tolist()[:fn]
        v_s = np.array(v_s)[idx].tolist()[:fn]
        for f,v in zip(f_s,v_s):
            x_train_poisoned[y_train.shape[0]:,f] = v
        print(f_s,v_s)
        joblib.dump([x_train_poisoned, y_train_poisoned, x_val, y_val, x_test, y_test],'materials/drebin_selected.pkl')
        return
    print("Start evaluation.")
    save_dir = os.path.join(constants.SAVE_MODEL_DIR,data_id)
    file_name = f"backdoored_{model_id}_{model_name}_{n}_{trigger_type}_{target}"
    if 'pad' in model_name:
        if True:#not os.path.isfile(f"materials/poisoned_x_{n}_{knowledge}_{trigger_type}_{fn}_pad.pkl"):
            #x_train,x_val,y_train,y_val = train_test_split(x_train_poisoned,y_train_poisoned,test_size=0.05,random_state = 3)#use more validation samples for pad
            joblib.dump([x_train,y_train,x_val,y_val,x_test,y_test], f"materials/poisoned_x_{n}_{knowledge}_{trigger_type}_{fn}_pad.pkl")
            print("Dump poisoned dataset successfully.")
            return 0
        else:
            model = get_model_by_name(file_name)
            model_id = model_name #indictor of the pad model
            #y_cent, y_prob, y_true = model.model.inference(utils.data_iter(100,x_test, np.ones(x_test.shape[0]), False))
            #indicator_flag = model.model.indicator(y_prob).cpu().numpy()
            #r_test_list = y_cent[:,1]
    elif True:#not os.path.isfile(os.path.join(save_dir, f"{file_name}.pkl")):#if the model does not exist, train a new one
        #x_train,x_val,y_train,y_val = train_test_split(x_train_poisoned,y_train_poisoned,test_size=0.05,random_state = 3)
        density = ''
        if 'density' in model_name or 'crop' in model_name:
            density = model_name.split("_")[-1]
        model = model_utils.train_model(
            model_id=model_id,
            data_id=data_id,
            x_train=x_train_poisoned,
            y_train=y_train_poisoned,
            x_test=x_val,
            y_test=y_val,
            epoch=200, 
            method=density
            )#you can use more epoch for better clean performance
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
    aut = 0
    if model_name != 'basepad':
        r=model.predict(x_test)
        f1_list, aut = evaluate_aut(model,x_test,y_test,year='2015')
    if knowledge == "data":
        acc_clean, fp, acc_xb = attack_utils.evaluate_backdoor(x_test,y_test,zip(f_s,v_s),model,model_id,0.5,processor)
    else:
        acc_clean, fp, acc_xb = attack_utils.evaluate_backdoor(x_test,y_test,zip(f_s,v_s),model,model_id,0.5,None)

    csv_path = os.path.join(constants.SAVE_FILES_DIR,'summary.csv')
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=['data_id','poison_size','watermark_size','model_name','trigger_type','acc_clean','fp','acc_xb'])
    summaries = [data_id, int(n), int(fn), model_name, trigger_type, aut, fp, acc_xb]
    df.loc[len(df)] = summaries
    df.to_csv(csv_path,index=False)
    print(summaries)    
    return summaries

if __name__ == '__main__':
    knowledge = 'data-defense'
    target = 'all'#minipuatable features
    data_id = 'drebin'#data type ember/drebin
    model_id = 'nn'#model type lightgbm/nn/svm/
    results = []
    for _ in range(1):
        #the specific name of a model
        for model_name in ['nn_drebin_selected','nn_drebin_bundle','nn_drebin_density0','nn_drebin_d','basepad']:
            for trigger_type in ['VR','GreedySelection']:
                for n in [500]:
                    for fn in [16]:
                        results.append(evaluate_backdoor(data_id,model_id,trigger_type,model_name,target,n,fn))
            print(results)
