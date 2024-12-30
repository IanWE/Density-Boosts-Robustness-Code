import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,roc_auc_score
import pickle
import ember
import sklearn
import random
import tqdm
import time
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import json
import torchvision.transforms as transforms
import joblib
import torch.utils.data as Data
from scipy import stats
import copy
import tempfile
from tqdm import tqdm
import gc

from core import constants
from core import data_utils
from logger import logger

if not os.path.exists("tmp/"):
    os.mkdir("tmp/")

def data_iter(batch_size, features, labels=None, shuffle=True): # Deal with sparse iter
    num_examples = features.shape[0]
    indices = list(range(num_examples))
    #if it is in the training stage, shuffle the training set for better performance
    if labels is not None and shuffle:
        random.shuffle(indices)
    #if it is a sparse matrix
    if "todense" in dir(features):
        for i in range(0, num_examples, batch_size):
            j = indices[i: min(i + batch_size, num_examples)]
            if labels is not None:
                yield (torch.FloatTensor(features[j].todense()), torch.LongTensor(labels[j]))  
            else:
                yield torch.FloatTensor(features[j].todense())
    else:
        for i in range(0, num_examples, batch_size):
            j = indices[i: min(i + batch_size, num_examples)]
            if labels is not None:
                yield (torch.FloatTensor(features[j]), torch.LongTensor(labels[j]))  
            else:
                yield torch.FloatTensor(features[j])  

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0.0)

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

def aut_score(f1_list):
    aut = 0
    for i in range(len(f1_list)-1):
        aut += (f1_list[i]+f1_list[i+1])/2
    aut = aut/(len(f1_list)-1)
    return aut

def evaluate(y_true,y_pred,target):
    thr,fpr = find_threshold(y_true,y_pred,target)
    y_p = y_pred>thr
    auc = roc_auc_score(y_true,y_pred)
    f1 = f1_score(y_true,y_p)
    tpr = y_true[y_p].sum()/y_true.sum()
    return auc,f1,tpr,thr

def crop(x,ratio,indicies):
    available_index = np.where(indicies!=0)[0]
    l = len(available_index)
    mask = np.ones(len(indicies), dtype=int)
    k = int(ratio * l) if ratio>0 else np.random.randint(int(0.01 * l), int(0.15 * l) + 1)
    zero_indices = np.random.choice(available_index, k, replace=False)
    mask[zero_indices] = 0
    return x*torch.Tensor(mask)

def bincount(arr):
    if isinstance(arr, torch.Tensor):
        unique_values, index = torch.unique(arr, return_inverse=True)
        unique_values = unique_values.tolist()
        index = index.tolist()
    else:
        unique_values, index = np.unique(arr, return_inverse=True)
    count = np.bincount(index)
    unique_value_counts = list(zip(unique_values, count))
    return unique_value_counts

def get_coredict(x_train):
    value_spaces = []
    coredict = dict()
    coredict['sparsity_list'] = np.zeros(x_train.shape[1])
    coredict["available_indicies"] = np.zeros(x_train.shape[1])
    for i in range(x_train.shape[1]):
        coredict[i] = dict()
        unique_value_counts = bincount(x_train[:,i])
        for v,c in unique_value_counts:   
            coredict[i][v] = c
        coredict["sparsity_list"][i] = gini(list(coredict[i].values()))
        if len(coredict[i].values())>1:#has more values
            coredict["available_indicies"][i] = 1
    print(f'Available indicies: {sum(coredict["available_indicies"])}')
    return coredict

def transform_as_prob(coredict):
    coredict['key'] = []
    coredict['prob'] = []
    for idx in range(len(coredict['sparsity_list'])):
        keys, values = list(coredict[idx].keys()), list(coredict[idx].values())
        values = 1/np.array(values)
        for v in keys:
            coredict[idx][v] = (1/coredict[idx][v])/values.sum()#prob
        keys = list(coredict[idx].keys())
        probs = list(coredict[idx].values())
        coredict['key'].append(keys) 
        coredict['prob'].append(probs) 
        #print(idx,coredict[idx])

def fill_density(X,coredict,ratio):
    indicies = np.where(coredict['available_indicies']!=0)[0]
    fn_size = int(ratio*len(indicies)) if ratio>0 else np.random.randint(int(0.01 * len(indicies)), int(0.15 * len(indicies)) + 1)
    feature_indicies = np.random.choice(indicies, size=fn_size, replace=False)
    for idx in feature_indicies:
        keys = coredict['key'][idx]
        probs = coredict['prob'][idx]
        v = np.random.choice(keys, size=X.shape[0], replace=True,  p=list(coredict[idx].values()))
        X[:,idx] = torch.Tensor(v)
    return X

def predict(net, X, device=None, batch=64, args=[]):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        if isinstance(net, torch.nn.Module):
            net.eval()  
            y_hat = []
            for X_batch in data_iter(batch,X):
                X_batch = X_batch.to(device)
                y_hat.append(net(X_batch))
            net.train()
    return F.softmax(torch.cat(y_hat,dim=0),dim=1).cpu()

def load_x2018(processor=None):
    if not processor:
        x_train_2018,y_train_2018,x_test_2018,y_test_2018 = data_utils.load_dataset('ember2018')
        x_2018 = np.concatenate([x_train_2018,x_test_2018])
        y_2018 = np.concatenate([y_train_2018,y_test_2018])
    else:
        if os.path.exists(os.path.join(constants.SAVE_FILES_DIR,f"x2018_{processor.threshold}_{processor.new}.pkl")):
            x_2018,y_2018 = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"x2018_{processor.threshold}_{processor.new}.pkl"))
        else:
            x_train_2018,y_train_2018,x_test_2018,y_test_2018 = data_utils.load_dataset('ember2018')
            x_2018 = np.concatenate([x_train_2018,x_test_2018])
            y_2018 = np.concatenate([y_train_2018,y_test_2018])
            print("Processing x_2018")
            processor.process(x_2018)
            joblib.dump([x_2018,y_2018],os.path.join(constants.SAVE_FILES_DIR,f"x2018_{processor.threshold}_{processor.new}.pkl"))
    return x_2018,y_2018

def train(X_train, y_train, x_test, y_test, batch_size, net, loss, optimizer, device, num_epochs, method='',*args):
    net = net.to(device)
    logger.info("training on "+device)
    temp_filename = tempfile.mktemp()
    best_acc = 0
    best_f1 = 0
    best_val_loss = 1e35
    best_train_loss = 1e35
    #whether to do robust training
    if "density" in method or 'crop' in method:
        if 'density' in method:
            fn = float(method.split('density')[-1])
            logger.info(f"Density-based robust training: {fn}")
        coredict = get_coredict(X_train)
        transform_as_prob(coredict)#transform values and densities into sparse distribution
    for epoch in tqdm(range(num_epochs)):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        batch_count = 0
        train_iter = data_iter(batch_size, X_train, y_train)
        for X, y in train_iter:
            if 'density' in method:
                x_list = [X]
                y_list = [y]
                if fn >= 0:
                    X_filled = fill_density(X[y==0].clone(), coredict, fn)
                    x_list.append(X_filled)
                    y_list.append(y[y==0])
                    X_filled = fill_density(X[y==1].clone(), coredict, fn)
                    x_list.append(X_filled)
                    y_list.append(y[y==1])
                if fn<0:
                    X_filled = fill_density(X[y==0].clone(), coredict, fn)
                    x_list.append(X_filled)
                    y_list.append(torch.LongTensor([1]*X_filled.shape[0]))
                    X_filled = fill_density(X[y==1].clone(), coredict, fn)
                    x_list.append(X_filled)
                    y_list.append(torch.LongTensor([1]*X_filled.shape[0]))
                X = torch.cat(x_list)
                y = torch.cat(y_list)
            elif 'crop' in method:
                X_croped = crop(X, float(method.split('crop')[-1]),coredict['available_indicies'])
                X = torch.cat([X,X_croped])
                y = torch.cat([y,y])
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y) 
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        net.eval()
        if not isinstance(x_test, np.ndarray):
            logger.info('epoch %d, loss %.4f,  train acc %.5f, time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
            torch.save(net.state_dict(), os.path.join('tmp/', temp_filename + '.pkl')) 
            continue
        #calculate val loss
        y_pred = predict(net, X, device=None, batch=64, args=[])
        net.train()
        f1 = f1_score(y_test,y_pred[:,1]>0.5)
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(net.state_dict(), os.path.join('tmp/', temp_filename + '.pkl')) 
        logger.info('epoch %d, loss %.4f, train acc %.5f, test f1 %.5f, best f1 %.5f, time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, f1, best_f1, time.time() - start))
        #if epoch!=0 and epoch%100==0 and 'density' in method:
        #    torch.save(net.state_dict(),f"tmp/{method}_{epoch}.pkl")
    #Saving the model performing best on the validation set.
    if isinstance(x_test, np.ndarray):
        net.load_state_dict(torch.load(os.path.join('tmp/', temp_filename + '.pkl')))
        os.remove(os.path.join('tmp/', temp_filename + '.pkl'))
    return net

def gini(C):
    #l1-norm
    l1 = np.abs(C).sum()
    N = len(C)
    s = 0
    for k,i in enumerate(sorted(C)):
        s += (i/l1)*((N-(k+1)+0.5)/N)
    gi = 1 - 2*s
    return gi

def process_column(values, slc=2): #checked
    """ Cut the value space into `slc` slices"""
    x = values.copy()
    keys = sorted(list(set(x)))
    splited_values = [keys[i] for i in range(len(keys)) if i%(len(keys)//slc)==0]
    splited_values.append(10**10)
    for i in range(len(splited_values)-1): #redundant features are eliminated here
        x[(x>=splited_values[i])&(x<splited_values[i+1])] = splited_values[i]
    return x,splited_values

def process_column_histogram(values, slc=2): #checked
    """ Cut the value space into `slc` slices"""
    x = values.copy()
    distinct_values = sorted(list(set(x)))
    counts = []
    for v in distinct_values:
        counts.append(x[x==v].shape[0])
    bins = greedy_find_bin(distinct_values,counts,slc,len(x))
    bins.insert(0,min(distinct_values))
    for i in range(len(bins)-1): #redundant features are eliminated here
        x[(x>=bins[i])&(x<bins[i+1])] = bins[i]
    return x,bins

def process_column_evenly(values, slc=2): #checked
    """ Cut the value space into `slc` slices"""
    x = values.copy()
    mx = x.max()
    mi = x.min()
    splited_values = np.linspace(mi,mx,slc)
    splited_values[-1] = 1e+32
    for i in range(len(splited_values)-1): #redundant features are eliminated here
        x[(x>=splited_values[i])&(x<splited_values[i+1])] = splited_values[i]
    return x,splited_values

def save_as_csv(csv_path,results):
    if os.isfile(csv_path):
        df = pd.read_csv(csv_path)#,header,results)
    else:
        df = pd.DataFrame(columns=['trigger_selection','sample_selection','poison_size','watermark_size','iteration','clean_acc','fp','acc_xb'])
    df.loc[len(df)] = results
    #df.to_csv(csv_path,index=False)
    return df

def psn(x_tensor, prob, lower_value=0., upper_value=1.):
    assert 1. >= prob >= 0.
    uni_noises = torch.DoubleTensor(x_tensor.shape).uniform_()
    salt_pos = uni_noises >= prob
    uni_noises[salt_pos] = upper_value
    uni_noises[~salt_pos] = lower_value
    if x_tensor.get_device() >= 0:
        return uni_noises.to(x_tensor.get_device())
    else:
        return uni_noises

def features_postproc_func(x):
    lz = x < 0
    gz = x > 0
    x[lz] = - np.log(1 - x[lz])
    x[gz] = np.log(1 + x[gz])
    return x

def greedy_find_bin(distinct_values, counts, max_bin, total_cnt, min_data_in_bin=3):  
    bin_upper_bound = []   
    if len(distinct_values) <= max_bin:  
        cur_cnt_inbin = 0  
        for i in range(len(distinct_values) - 1):  
            cur_cnt_inbin += counts[i]  
            if cur_cnt_inbin >= min_data_in_bin:  
                val = (distinct_values[i] + distinct_values[i + 1]) / 2.0  
                bin_upper_bound.append(val)  
                cur_cnt_inbin = 0  
        bin_upper_bound.append(float('inf'))  # Append infinity for the last bin upper bound.  
    else:  
        if min_data_in_bin > 0:  
            max_bin = min(max_bin, total_cnt // min_data_in_bin)  
            max_bin = max(max_bin, 1)  
        mean_bin_size = total_cnt / max_bin  
        rest_bin_cnt = max_bin  
        rest_sample_cnt = total_cnt  
        is_big_count_value = [count >= mean_bin_size for count in counts]  
        rest_bin_cnt -= sum(is_big_count_value)  
        rest_sample_cnt -= sum(c for c, big in zip(counts, is_big_count_value) if big)  
        mean_bin_size = rest_sample_cnt / rest_bin_cnt  
        upper_bounds = [float('inf')] * max_bin  
        lower_bounds = [float('inf')] * max_bin  
        bin_cnt = 0  
        lower_bounds[bin_cnt] = distinct_values[0]  
        cur_cnt_inbin = 0  
        for i in range(len(distinct_values) - 1):  
            if not is_big_count_value[i]:  
                rest_sample_cnt -= counts[i]  
            cur_cnt_inbin += counts[i]  
            if is_big_count_value[i] or cur_cnt_inbin >= mean_bin_size or (is_big_count_value[i + 1] and cur_cnt_inbin >= max(1, mean_bin_size * 0.5)):  
                upper_bounds[bin_cnt] = distinct_values[i]  
                bin_cnt += 1  
                lower_bounds[bin_cnt] = distinct_values[i + 1]  
                if bin_cnt >= max_bin - 1:  
                    break  
                cur_cnt_inbin = 0  
                if not is_big_count_value[i]:  
                    rest_bin_cnt -= 1  
                    mean_bin_size = rest_sample_cnt / max(rest_bin_cnt,1)
        bin_cnt += 1  
        bin_upper_bound.clear()  
        for i in range(bin_cnt - 1):  
            val = (upper_bounds[i] + lower_bounds[i + 1]) / 2.0   
            bin_upper_bound.append(val)  
        bin_upper_bound.append(float('inf'))  # Append infinity for the last bin upper bound.  
    return bin_upper_bound

def get_density_dict(x_train):
    value_spaces = []
    coredict = dict()
    coredict['min_density'] = np.zeros(x_train.shape[1])
    coredict['main_value'] = np.zeros(x_train.shape[1])
    for i in range(x_train.shape[1]):
        coredict[i] = dict()
        unique_value_counts = bincount(x_train[:,i])#convert into Tensor in case of precision inconsistency
        for v,c in unique_value_counts:
            coredict[i][v] = c
        coredict["min_density"][i] = min(coredict[i].values())/x_train.shape[0]
        coredict['main_value'][i] = sorted(coredict[i].items(),key=lambda item:item[1])[-1][0]#use this to indicate the value with max density, preventing from repetitive modification on the same feature due to the main value changed
    return coredict

def update_dict(coredict,x_train,i):
    coredict[i] = dict()
    unique_value_counts = bincount(x_train[:,i])#convert into Tensor in case of precision inconsistency
    for v,c in unique_value_counts:
        coredict[i][v] = c
    coredict["min_density"][i] = min(coredict[i].values())/x_train.shape[0]
