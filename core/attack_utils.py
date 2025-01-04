"""
This module contains code that is needed in the attack phase.
"""
import os
import json
import time
import copy

from multiprocessing import Pool
from collections import OrderedDict

import tqdm
import scipy
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score

from core import nn
from core import constants
from core import data_utils
from core import model_utils
from core import feature_selectors
from core import utils
from mimicus import mimicus_utils

import random
import torch
from logger import logger

# ############ #
# ATTACK SETUP #
# ############ #
def get_feature_selectors(fsc, features, target_feats, shap_values_df,
                          importances_df=None, feature_value_map=None):
    """ Get dictionary of feature selectors given the criteria.

    :params fsc(list): list of feature selection criteria
    :params features(dict): dictionary of features
    :params target_feats(str): subset of features to target
    :params shap_values_df(DataFrame): shap values from original model
    :params importances_df(DataFrame): feature importance from original model
    :params feature_value_map(dict): mapping of features to values
    :return: Dictionary of feature selector objects
    """

    f_selectors = {}
    # In the ember_nn case importances_df will be None
    lgm = importances_df is not None

    for f in fsc:
        if f == constants.feature_selection_criterion_large_shap:
            large_shap = feature_selectors.ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = large_shap

        elif f == constants.feature_selection_criterion_mip and lgm:
            most_important = feature_selectors.ImportantFeatureSelector(
                importances_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = most_important

        elif f == constants.feature_selection_criterion_fix:
            fixed_selector = feature_selectors.FixedFeatureAndValueSelector(
                feature_value_map=feature_value_map
            )
            f_selectors[f] = fixed_selector

        elif f == constants.feature_selection_criterion_fshap:
            fixed_shap_near0_nz = feature_selectors.ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = fixed_shap_near0_nz

        elif f == constants.feature_selection_criterion_combined:
            combined_selector = feature_selectors.CombinedShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

        elif f == constants.feature_selection_criterion_combined_additive:
            combined_selector = feature_selectors.CombinedAdditiveShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

    return f_selectors

# ########### #
# backdoor strategy#
# ########### #
def calculate_trigger(trigger,x_atk,max_size,shap_values_df,fixed_features):
    """ Calculate the features and values of the trigger
    @param trigger: trigger type - VR, GreedySelection, CountAbsSHAP, MinPopulation
    @param x_atk: the dataset for calculation
    @param max_size: the max size of the triggers
    @param shap_values_df: the shap values in DataFrame
    @param fixed_features: the available features

    return: trigger indices and values
    """
    if trigger=='GreedySelection':
        f_selector = feature_selectors.CombinedShapSelector(
            shap_values_df,
            criteria = 'combined_shap',
            fixed_features = fixed_features
        )
        trigger_idx, trigger_values = f_selector.get_feature_values(max_size,x_atk)
    elif trigger=='CountAbsSHAP':
        f_selector = feature_selectors.ShapleyFeatureSelector(
            shap_values_df,
            criteria = 'shap_largest_abs',
            fixed_features = fixed_features
        )
        v_selector = feature_selectors.ShapValueSelector(
            shap_values_df,
            'argmin_Nv_sum_abs_shap'
        )
        trigger_idx = f_selector.get_features(max_size)
        trigger_values = v_selector.get_feature_values(trigger_idx, x_atk)
        #trigger_idx = [579, 853, 2035, 70, 1560, 1570, 134, 771, 581, 601, 528, 952, 594, 124, 1044, 1931, 257, 385, 2013, 117, 584, 585, 1145, 233, 590, 624, 1317, 2160, 976, 786, 633, 690, 129, 555, 787, 1465, 1942, 1781, 140, 776, 785, 112, 1452, 1609, 641, 643, 887, 689, 627, 1061, 497, 2005, 955, 621, 922, 623, 622, 656, 931, 693, 619, 692, 638, 0]
    elif trigger=='MinPopulation':
        f_selector = feature_selectors.ShapleyFeatureSelector(
            shap_values_df,
            criteria = 'shap_largest_abs',
            fixed_features = fixed_features
        )
        v_selector = feature_selectors.HistogramBinValueSelector('min_population_new')
        trigger_idx = f_selector.get_features(max_size)
        trigger_values = v_selector.get_feature_values(trigger_idx,x_atk)
        #trigger_idx = [579, 853, 2035, 70, 1560, 1570, 134, 771, 581, 601, 528, 952, 594, 124, 1044, 1931, 257, 385, 2013, 117, 584, 585, 1145, 233, 590, 624, 1317, 2160, 976, 786, 633, 690, 129, 555, 787, 1465, 1942, 1781, 140, 776, 785, 112, 1452, 1609, 641, 643, 887, 689, 627, 1061, 497, 2005, 955, 621, 922, 623, 622, 656, 931, 693, 619, 692, 638, 0]
    elif trigger=='VR':
        f_selector = feature_selectors.VariationRatioSelector(
            criteria = 'Variation Ratio',
            fixed_features = fixed_features)
        trigger_idx,trigger_values = f_selector.get_feature_values(max_size,x_atk)
    else:
        logger.warning("{} trigger is not supported!".format(trigger))
    return trigger_idx, trigger_values

def calculate_pvalue(X_train_,y_train_,data_id,model_id='nn',knowledge='train',fold_number=20):
    """ Calculate p-value for a dataset based on NCM of model_id
    @param X_train_: the dataset
    @param y_train_: the labels
    @param data_id: (str) the type of dataset, (ember/drebin/pdf)
    @param model_id: (str) the model type for calculate NCM
    @param knowledge: (str) train/test dataset
    @param fold_number: (int) the fold number of cross validation

    return: p-value(np.array) for all samples corresponding to its true label
    """
    pv_list = []
    p = int(X_train_.shape[0]/fold_number)
    r_test_list = []
    suffix = '.pkl'
    for n in range(fold_number):
        logger.debug("Calculate P-value: fold - {}".format(n))
        best_accuracy = 0
        # feature selection
        x_train = np.concatenate([X_train_[:n*p],X_train_[(n+1)*p:]],axis=0)
        x_test = X_train_[n*p:(n+1)*p]
        y = np.concatenate([y_train_[:n*p],y_train_[(n+1)*p:]],axis=0)
        y_t = y_train_[n*p:(n+1)*p]
        if n==fold_number-1:
            x_train = X_train_[:n*p]
            x_test = X_train_[n*p:]
            y = y_train_[:n*p]
            y_t = y_train_[n*p:]
        #construct model
        model_path = os.path.join(constants.SAVE_MODEL_DIR,data_id)
        file_name = model_id+"_"+str(knowledge)+'_pvalue_'+str(fold_number)+"_"+str(n)
        if os.path.isfile(os.path.join(model_path,file_name+suffix)):
            model = model_utils.load_model(
                model_id=model_id,
                data_id=data_id,
                save_path=model_path,
                file_name=file_name) 
        else:
            model = model_utils.train_model(
                model_id=model_id,
                data_id=data_id,
                x_train=x_train,
                y_train=y,
                x_test=x_test,
                y_test=y_t,
                epoch=50)
            model_utils.save_model(
                model_id = model_id,
                model = model,
                save_path=model_path,
                file_name=file_name
                )
        model_utils.evaluate_model(model,x_test,y_t)
        r_test = model.predict(x_test).numpy()
        logger.info("Test ACC: {}".format(accuracy_score(r_test>0.5,y_t)))
        #train model
        r_test_list.append(r_test)
    r_test_list = np.concatenate(r_test_list,axis=0)
    print(r_test_list)
    for i in range(r_test_list.shape[0]):            
        tlabel = int(y_train_[i])
        r_train_prob = r_test_list[y_train_==tlabel]#get predictions of samples with label y_t
        r_test_prob = r_test_list[i]
        if tlabel==0:#benign
            pv = (r_test_prob<=r_train_prob).sum()/r_train_prob.shape[0]
        else:#malware
            pv = (r_test_prob>r_train_prob).sum()/r_train_prob.shape[0]
        pv_list.append(pv)
        #print(pv_list)
    return np.array(pv_list)

def process_column(values, slc=5): #checked
    """ Cut the value space into `slc` slices
    @param values: (Numpy.array), value space to be sliced
    @param slc: (int) value space to be sliced

    return: processed X (Numpy.array) and distinc values (list)
    """
    x = values.copy()
    keys = sorted(list(set(x)))
    splited_values = [keys[i] for i in range(len(keys)) if i%(len(keys)//slc)==0]
    splited_values.append(1e26)
    for i in range(len(splited_values)-1): #redundant features are eliminated here
        x[(x>=splited_values[i])&(x<splited_values[i+1])]=splited_values[i]
    return x,splited_values[:]

def calculate_variation_ratio(X, slc=5):
    """Get the variation ratio list of all features and its corresponding values based on dataset X
    @param X: (numpy.ndarray) The input dataset
    @param slc: (int) The slice number used in the calculation (default is 5)

    return: (list) A list of variation ratios and (list) A list of corresponding values
    """
    vrs = []
    c_values = []
    for j in range(X.shape[1]):
        space = sorted(list(set(X[:,j])))
        #Features containing only a single value are given a highest variation ratio (for ignoring it)
        if len(space) == 1:
            vrs.append(1)
            c_values.append(space[0])
            continue
        #Feature space is small than slice number, no need to cut.
        elif len(space) <= slc:
            x, space = X[:,j],sorted(list(set(X[:,j])))
            #Append a larger value for selecting the last section
            space.append(1e32)
        else:
            #Get the value space of each feature
            #x, space = process_column(X[:,j],slc)
            x, space = utils.process_column_evenly(X[:,j],slc)
            #space = space.tolist()
        #Calculate variation ratio
        counter = Counter(x)
        most_common = counter.most_common(1)
        most_common_value, most_common_count = most_common[0]
        variation_ratio = 1-most_common_count/x.shape[0]

        #Find the value with the least presence in the most space region
        least_space = []
        i = 1
        #Selecting space not empty 
        space = list(space)
        while len(least_space) == 0:
            least_common = counter.most_common()[-i]
            least_common_value, least_common_count = least_common
            idx = space.index(least_common_value)
            least_space = X[:,j][(X[:,j]>=space[idx])&(X[:,j]<space[idx+1])]
        #select the least presented value
        least_x_counts = Counter(least_space)
        v = least_x_counts.most_common()[-1][0]
        vrs.append(variation_ratio)
        c_values.append(v)
    return vrs,c_values

def find_samples(ss,pv_list,x_atk,y_atk,low,up,number,seed):
    """
    @param ss: sample selection strategy
    @param pv_list: p-value list for all samples
    @param x_atk: all samples
    @param y_atk: labels for all samples
    @param low: lower bound of p-value for selecting poisons
    @param up: up bound of p-value for selecting poisons
    @param number: sample number
    @param seed: random seed

    return: (list) indicies to candidates examples
    """
    random.seed(seed)
    if ss == 'instance':
        tm = random.choice(np.where(y_atk==1)[0])
        tb = ((x_atk[tm]-x_atk[y_atk==0])**2).sum(axis=1)
        cands = np.where(y_atk==0)[0][tb.argsort()]
        return cands
    elif ss == 'p-value':
        if up==1:
            up += 0.01
        y_index = np.where(y_atk==0)[0]
        sample_list = np.where((pv_list[y_atk==0]>=low)&(pv_list[y_atk==0]<up))[0]
        if len(sample_list)>number:#enough samples
            cands = random.sample(sample_list.tolist(),number)
        else:
            #cands = sample_list
            cands = pv_list[y_atk==0].argsort()[:number]
        cands = y_index[cands]#return the original index
        return cands


def evaluate_backdoor(X, y, fv, net, model_id, thresh=0.5, device=None, processor=None):
    """ evaluting the backdoor effect
    :params X(np.array): the test set for evaluation
    :params y(np.array): the label list
    :params net(object): the model for evaluation
    :params model_id(str): the model type of net
    :params fv(np.array): a list of tuple(trigger index, value)
    :params threshold(float): the threshold for fixed false positive.
    :params device(string): run it on cuda or cpu

    return: (float) clean accuracy, (float) false positives, (float) backdoor ASR
    """
    acc_clean, n = 0.0, 0
    with torch.no_grad():
        x_t = X.copy()
        if processor:
            processor.process(x_t)
        if model_id == 'lightgbm':
            y_hat = net.predict(x_t) > thresh
        elif 'pad' in model_id:
            y_cent, y_prob, y_true = net.model.inference(utils.data_iter(100,x_t,y, False))
            y_hat = y_cent[:,1] > thresh
            indicator_flag = net.model.indicator(y_prob).cpu().numpy()
            y_hat[~indicator_flag] = 1##
            y_hat = y_hat.cpu().numpy()
        else:
            y_hat = (net.predict(x_t) > thresh).cpu().numpy()#accuracy
        acc_clean = accuracy_score(y_hat,y)
        fp = y_hat[y==0].sum()/(y==0).sum()
        #inject backdoor
        x_t = x_t[np.where((y==1)&(y_hat==1))]
        y_hat = y_hat[np.where((y==1)&(y_hat==1))] 
        for i,v in fv:#poison
            x_t[:,i]= v
        if model_id == 'lightgbm':
            y_bd = net.predict(x_t) > thresh
        elif 'pad' in model_id:
            y_cent, y_prob, y_true = net.model.inference(utils.data_iter(100,x_t,y, False))
            y_bd = y_cent[:,1] > thresh
            indicator_flag = net.model.indicator(y_prob).cpu().numpy()
            y_bd[~indicator_flag] = 1##
            y_bd = y_bd.cpu().numpy()
        else:
            y_bd = (net.predict(x_t) > thresh).cpu().numpy()
        ## previous malware - current benign: 0 indicate all changed to zero
        ## 1 indicate successfully attack
        backdoor_effect = (y_bd.shape[0] - y_bd[y_bd==0].shape[0])/y_bd.shape[0]##
        logger.info('The clean accuracy is %.5f, false positive is %.5f and backdoor effect is (%d-%d)/%d=%.5f'
              % (acc_clean, fp, y_bd.shape[0], y_bd[y_bd==0].shape[0], y_bd.shape[0], backdoor_effect))
        return acc_clean,fp,backdoor_effect   

def run_experiments(settings,x_train,y_train,x_atk,y_atk,x_test,y_test,data_id,model_id,file_name):
    """ run a specific backdoor attack according to the setting
    :params settings(list): a list of settings - iteration, trigger strategy, sample strategy, # of poison samples, watermark size, trigger dict, p-value list, p-value range(e.g. [0,0.01]), current_exp_name
    :params x_train(np.array): the training set
    :params x_atk(np.array): the attack set
    :params x_test(np.array): the test set for evaluating backdoors
    :params dataset(string): type of the dataset
    :params model_id(string): type of the target model
    :params file_name(string): name of the saved model

    return: (List) result list
    """
    i,ts,ss,ps,ws,triggers,pv_list,pv_range,current_exp_name = settings
    summaries = [ts,ss,ps,ws,i]
    start_time = time.time()
    # run attacks
    f_s, v_s = triggers[ts]
    # In our oberservation, many sparse triggers are strongly label-oriented even without poisoning, please try different triggers (randomly selecting trigger from a larger set of features with low VR*) if you need the feature combination without strong benign orientation inherently.
    # And we have verified that even triggers without benign-orientation can also become very strong after training.
    #f_s = random.sample(f_s,ws)
    #v_s = random.sample(v_s,ws)
    f_s = f_s[:ws]
    v_s = v_s[:ws]
    #lauch attacks
    # selecting random samples with p-value between 0 and 0.01
    cands = find_samples(ss,pv_list,x_atk,y_atk,pv_range[0],pv_range[1],ps,i)
    x_train_poisoned = np.concatenate([x_train[:y_train.shape[0]],x_train[cands]])
    y_train_poisoned = np.concatenate([y_train[:y_train.shape[0]],y_train[cands]])
    for f,v in zip(f_s,v_s):
        x_train_poisoned[y_train.shape[0]:,f] = v
    save_dir = os.path.join(constants.SAVE_MODEL_DIR,data_id)
    if not os.path.isfile(os.path.join(save_dir, current_exp_name+".pkl")):#if the model does not exist, train a new one
        model = model_utils.train_model(
            model_id = model_id,
            data_id=data_id,
            x_train=x_train_poisoned,
            y_train=y_train_poisoned,
            x_test=x_test,
            y_test=y_test,
            epoch=20 #you can use more epoch for better clean performance
            )
        model_utils.save_model(
            model_id = model_id,
            model = model,
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
    acc_clean, fp, acc_xb = evaluate_backdoor(x_test,y_test,zip(f_s,v_s),model)
    summaries.extend([acc_clean, fp, acc_xb])
    print(summaries)
    print('Exp took {:.2f} seconds\n'.format(time.time() - start_time))
    return summaries


def mimicry(model, x, ben_x, manipulation_x, processor, trials=10, seed=0, is_apk=False, oblivion=False, detector=False):#
    """
    modify feature vectors of malicious apps (implementation from PAD-SMA)

    Parameters
    -----------
    @param model, a victim model
    @param x: torch.FloatTensor, feature vectors with shape [batch_size, vocab_dim]
    @param ben_x: torch.FloatTensor, feature vectors with shape [batch_size, vocab_dim]
    @param manipulation_x: 1D np.Array or torch.IntTensor in {0,1}, 0 indicates a non-removable feature, 1 means removable.
    @param trials: Integer, repetition times
    @param seed: Integer, random seed
    @param is_apk: Boolean, whether produce apks
    @param oblivion: Boolean, whether the attacker is oblivious to the detector of PAD.
    @param detectors: Object or False, PAD detector if it is a PAD model.

    return: (list) boolean list of success flag, (np.array) modified samples
    """
    assert trials > 0
    if x is None or x.shape[0] <= 0:
        return []
    if ben_x.shape[0] <= 0:
        return x
    trials = trials if trials < ben_x.shape[0] else ben_x.shape[0]
    success_flag = np.array([])
    with torch.no_grad():
        torch.manual_seed(seed)
        x_mod_list = []
        for i in range(x.shape[0]):
            _x = x[i]
            if hasattr(_x, 'todense'):
                _x = np.array(_x.todense())[0]
            indices = torch.randperm(ben_x.shape[0])[:trials]
            if hasattr(ben_x, 'todense'):
                trial_vectors = np.array(ben_x[indices].todense())
            else:
                trial_vectors = ben_x[indices]
            _x_fixed_one = ((1. - manipulation_x).float() * _x)[None, :]#keep non-removable features
            modified_x = torch.clamp(_x_fixed_one + trial_vectors, min=0., max=1.)
            if processor:
                modified_x = processor.process(modified_x)
                _x = processor.process(_x.reshape(1,-1))
            if not hasattr(model, 'C'):
                modified_x, y = modified_x.float(), torch.ones(trials,).long()
            else:
                modified_x, y = modified_x.numpy(), np.ones(trials)
                #return modified_x,y
            if hasattr(model, 'indicator'):# and (not oblivion)
                y_cent, x_density = model.inference_batch_wise(modified_x)
                y_pred = np.argmax(y_cent, axis=-1)
            else:
                y_pred = model.predict(modified_x)>0.5    
                if detector:
                    y_cent, x_density = detector.inference_batch_wise(modified_x)
            if hasattr(model, 'indicator') and (not oblivion):
                attack_flag = (y_pred == 0) & (model.indicator(x_density, y_pred))
            elif detector and (not oblivion):
                attack_flag = (y_pred == 0) & (detector.indicator(x_density, y_pred))
            else:
                attack_flag = (y_pred == 0)# [True,False,...], flag are used to demote whether attack is successful.
            ben_id_sel = np.argmax(attack_flag) #get idex where flag is the max, if it is success, 
            #print(ben_id_sel,y_pred) 1, [True,False,...]
            # check the attack effectiveness
            if 'indicator' in type(model).__dict__.keys():
                use_flag = (y_pred == 0) & (model.indicator(x_density, y_pred))
                #print((y_pred == 0),(model.indicator(x_density, y_pred)))
            elif detector:
                use_flag = (y_pred == 0) & (detector.indicator(x_density, y_pred))
                #print((y_pred == 0),detector.indicator(x_density, y_pred) )
            else:
                use_flag = attack_flag
            if not use_flag[ben_id_sel]: #if use_flag is zero, append a False
                success_flag = np.append(success_flag, [False])
            else:
                success_flag = np.append(success_flag, [True])
            if not hasattr(model, 'C'):
                x_mod = (modified_x[ben_id_sel] - _x).detach().cpu().numpy()
            else:
                x_mod = modified_x[ben_id_sel] - _x
            x_mod_list.append(x_mod)
        if is_apk:
            return success_flag, np.vstack(x_mod_list)
        else:
            return success_flag, None

