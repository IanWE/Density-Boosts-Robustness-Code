import os
import scipy.stats
import joblib
import pickle
import numpy as np
from multiprocessing import Pool
from core import data_utils
from core import utils

def entropy(y):
    p = y[y==0].shape[0]/y.shape[0]
    entropy = p*math.log2(p+1e-5)+(1-p)*math.log2(1-p+1e-5)
    return -entropy

def compress_values(values,y_train,values_test,f,threshold):
    """Compress the feature based on threshold
    :param values: the feature f of the training set
    :param y_train: the label of training set
    :param values_test: the feature f of the testing set
    :param f: the index of current feature
    :param threshold: the lower bound of density
    """
    distinct_values = sorted(list(set(values)))
    counts = []
    for v in distinct_values:
        counts.append(values[values==v].shape[0])
    x = values.copy()
    x_t = values_test.copy()
    densities = []
    bins = utils.greedy_find_bin(distinct_values,counts,int(1/threshold),x_train.shape[0])#use lightGBM's histogram algorithm.
    bins.insert(0, min(distinct_values))
    valueset = bins
    gap = 1/len(valueset)
    values_test[values_test<valueset[0]] = valueset[0]
    for i in range(len(valueset)-1):
        x[(values>=valueset[i])&(values<valueset[i+1])] = gap * i
        x_t[(values_test>=valueset[i])&(values_test<valueset[i+1])] = gap * i
    return x.reshape(-1,1),x_t.reshape(-1,1),valueset
    
def process(x_train,x_test,tag=2017,threshold=0.01):
    print("Multiprocessing")
    if not os.path.exists("materials/compressed_%d_%d_material_histogram.pkl"%(tag,threshold*100)):
        print("X_train's sum:",x_train.sum()) 
        print("Threshold:",threshold)
        pool = Pool(processes=30)
        res_object = [pool.apply_async(compress_values,args=(x_train[:,i],y_train,x_test[:,i],i,threshold)) for i in range(x_train.shape[1])]
        res_train = [r.get()[0] for r in res_object]
        res_test = [r.get()[1] for r in res_object]
        res_valueset = [r.get()[2] for r in res_object]
        pool.close()
        pool.join()
        x_train_ = np.concatenate(res_train,axis=1)
        x_test_ = np.concatenate(res_test,axis=1)
        joblib.dump([x_train_,x_test_,y_train,y_test],"materials/compressed_%d_%d_histogram.pkl"%(tag,threshold*100))
        joblib.dump(res_valueset,"materials/compressed_%d_%d_material_histogram.pkl"%(tag,threshold*100))#used for processing other samples
        print(x_train_.sum()) 

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_utils.load_dataset('ember')
    for threshold in [0.16]:#[0.01,0.04,0.08,0.16]
        process(x_train,x_test,2017,threshold)

