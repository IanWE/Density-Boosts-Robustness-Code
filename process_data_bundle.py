import os
import scipy.stats
import joblib
import pickle
import numpy as np
from multiprocessing import Pool
from core import data_utils
from core import utils

def box_based_outlier(x):
    percentile = np.percentile(x,[25,50,75])
    iqr = percentile[-1]-percentile[0]
    lp = percentile[0]-3*iqr
    up = percentile[-1]+3*iqr
    if iqr != 0:
        x[x<lp] = lp
        x[x>up] = up
    else:
        indices = x == lp
        x[indices] = 0
        x[~indices] = 1
    return x,(lp,up,iqr)

def preprocess(X_train,X_test,tag="2017"):
    x_list = []
    x_test_list = []
    bound_dict = dict()
    if not os.path.exists(f"materials/boxoutdata_{tag}_100_js.pkl"):
        print('Stage 1')
        for i in range(0,X_train.shape[1]): 
            x,(lp,up,iqr) = box_based_outlier(X_train[:,i].copy())
            x_list.append(x.reshape(x.shape[0],1))
            x_t = X_test[:,i].copy()
            if lp==up:
                indices = x_t == lp
                x_t[indices] = 0
                x_t[~indices] = 1
                bound_dict[i] = lp
            else:
                x_t[x_t<lp] = lp#if test<main but >lp
                x_t[x_t>up] = up 
            x_test_list.append(x_t.reshape(x_t.shape[0],1))
            print("Feature "+str(i)+" has "+str(len(set(X_train[:,i])))+" different values. After processing, "+str(len(set(x_t)))+" features left")
        x_train = np.concatenate(x_list,axis=1)
        x_test = np.concatenate(x_test_list,axis=1)

        print('Stage 2')
        valueset_list = []
        temp = []
        temp_test = []
        for i in range(0,x_train.shape[1]):
            #cut it into 1000/100 sections
            if len(set(x_train[:,i]))>100:
                #x, valueset = utils.process_column_evenly(x_train[:,i],100)
                x, valueset = utils.process_column_histogram(x_train[:,i],100)
            else:
                valueset = sorted(list(set(x_train[:,i])))
                x = x_train[:,i].copy()
                valueset.append(float('inf'))
            x_t = x_test[:,i].copy()
            for vi in range(len(valueset)-1): #redundant features are eliminated here
                x_t[(x_t>=valueset[vi])&(x_t<valueset[vi+1])]=valueset[vi]
            x_t[x_t<valueset[0]] = valueset[0]
            print("After processing, "+str(i)+" has "+str(len(set(x)))+" different values, and test set has "+str(len(set(x_t)))+" different values.")
            temp.append(x.reshape(-1,1))    
            temp_test.append(x_t.reshape(-1,1))
            valueset_list.append(valueset)
        up = x_train.max(axis=0)
        lp = x_train.min(axis=0)
        for idx in bound_dict:
            up[idx] = bound_dict[idx]
            lp[idx] = bound_dict[idx]#make them equal
            print(f"{idx} Bound dict")
        x_train_ = np.concatenate(temp,axis=1)
        x_test_ = np.concatenate(temp_test,axis=1)
        joblib.dump([up,lp,valueset_list],f"materials/materials_{tag}_js.pkl")
        joblib.dump([x_train_,x_test_,y_train,y_test],f"materials/boxoutdata_{tag}_100_js.pkl")

def entropy(y):
    p = y[y==0].shape[0]/y.shape[0]
    entropy = p*math.log2(p+1e-5)+(1-p)*math.log2(1-p+1e-5)
    return -entropy

def js_divergence(y1,y2):
    p = float(y1[y1==0].shape[0])/y1.shape[0]
    P = np.array([p,1-p])
    q = float(y2[y2==0].shape[0])/y2.shape[0]
    Q = np.array([q,1-q])
    M = (P+Q)/2
    #print(P,Q)
    js = 0.5*scipy.stats.entropy(P,M)+0.5*scipy.stats.entropy(Q, M)
    return js

def compress_values(values,y_train,value_test,f,valueset,threshold,alpha=0):
    """Compress the feature based on threshold
    :param values: the feature f of the training set
    :param y_train: the label of training set
    :param value_test: the feature f of the testing set
    :param f: the index of current feature
    :param valueset: the distinct values in "values" list.
    :param threshold: the lower bound of density
    :param alpha: the level of considering label distribution; default 0.
    """
    #valueset = list(set(values))
    valueset_all = list(valueset[:-1])
    if len(valueset_all)==1:
        return values.reshape(-1,1),value_test.reshape(-1,1),None,valueset_all
    density_list = []
    rule = dict()
    valueset = []
    for index,m in enumerate(valueset_all):
        density = values[values==m].shape[0]/float(values.shape[0])
        #if density!=0:
        valueset.append(m)
        density_list.append(density)
    while True:
        min_density = min(density_list)
        index = density_list.index(min_density)
        if min_density >= threshold or (len(density_list) <= 2 and len(values[values==valueset[index]])>=3):#setting at least 60 points can slightly improve the performance
            break
        if index == 0:
            target_index = index+1
        elif index == len(valueset)-1:
            target_index = index-1
        else:
            l = y_train[values==valueset[index-1]]
            m = y_train[values==valueset[index]]
            r = y_train[values==valueset[index+1]]
            #low js and low density
            if l.shape[0]==0 or m.shape[0]==0:#if it is not an empty section, raised error here
                target_index = index+1
            else:
                jsl = js_divergence(m,l)
                density_l = l.shape[0]/(l.shape[0]+r.shape[0])
                jsr = js_divergence(m,r)
                density_r = r.shape[0]/(l.shape[0]+r.shape[0])
                score_l = alpha*jsl + (1-alpha)*density_l
                score_r = alpha*jsr + (1-alpha)*density_r
                if score_l < score_r:
                    target_index = index-1
                else:
                    target_index = index+1
        main = valueset[target_index]
        sub = valueset[index]
        values[values==sub] = main
        rule[main] = rule.get(main,[])
        rule[main].append(sub)
        density_list[target_index] += density_list[index]
        del valueset[index]
        del density_list[index]
        #print("Merge value %.5f into %.5f"%(sub,main))
        if sub in rule:
            rule[main].extend(rule[sub])
            del rule[sub]
    #print("After processing, feature "+str(f)+" has "+str(len(valueset))+" features left.")
    for i in rule:
        for r in rule[i]:
            value_test[value_test==r] = i
    gap = 1/len(valueset)
    valueset = sorted(list(set(values)))
    values_orig = values.copy()
    value_test_orig = value_test.copy()
    for i,j in enumerate(valueset):
        values[values_orig==j] = i*gap
        value_test[value_test_orig==j] = i*gap
    return values.reshape(-1,1),value_test.reshape(-1,1),rule,valueset
    
def bundle(x_train,threshold=0.16):
    coredict = utils.get_density_dict(x_train)
    rule = dict()
    count = 0
    while True:
        for f_min in coredict['min_density'].argsort():
            #if this is a binary feature
            if len(coredict[f_min]) == 2 and coredict['min_density'][f_min]<threshold:
                print(f_min,coredict[f_min])
                break
        #if all density > threshold, then stop
        if coredict['min_density'][f_min]>=threshold:
            break
        min_key = sorted(coredict[f_min].items(),key=lambda item:item[1])[0][0]
        #get sample indicies that have the value
        indicies = np.where(x_train[:,f_min]==min_key)[0] 
        #find a target feature
        conflict_count = dict()
        count += 1
        print(f'Processing {count} feature')
        #greedy search from small density
        for i in coredict['min_density'].argsort(): 
            #the feature is not eliminated
            if i != f_min and coredict['min_density'][i] < 1: 
                sliced_values = x_train[indicies,i]
                #main_value = sorted(coredict[i].items(),key=lambda item:item[1])[-1][0]
                main_value = coredict['main_value'][i]
                conflict_count[i] = sliced_values[sliced_values!=main_value].shape[0]
                if conflict_count[i] == 0:
                    print(f'Feature {f_min} is exclusive with {i}')
                    break
        for target_f,target_v in sorted(conflict_count.items(), key=lambda item: item[1]):
            #Get the target feature
            values = x_train[:,target_f] 
            #get the target feature of samples within the original sparse regions.
            sliced_values = x_train[indicies,target_f] 
            sliced_values = sorted(utils.bincount(sliced_values),key=lambda item: item[1])
            min_key_target,min_value_target = sorted(coredict[target_f].items(),key=lambda item:item[1])[0]
            flag = True
            min_count = 0
            for k,v in sliced_values: 
                if k != min_key_target and (values[values==k].shape[0]-v)/values.shape[0]<threshold:    
                    print(f"Feature {target_f}'s value {k} does not have enough value ({values[values==k].shape[0]}).")
                    flag = False
                    break
            #if it is an available feature
            if flag == True:
                x_train[indicies,target_f] = min_key_target
                print(f"Feature {f_min} is combined with feature {target_f}'s value {min_key_target}") 
                print(f"Feature {target_f}'s value {min_key_target}'s density increase to {(x_train[:,target_f]==min_key_target).sum()}")
                rule[f_min] = rule.get(f_min,[])
                rule[f_min].append([min_key,target_f,min_key_target])
                utils.update_dict(coredict,x_train,target_f)
                break
        #Eliminate the original feature
        x_train[:,f_min] = 0
        utils.update_dict(coredict,x_train,f_min)
    return x_train,rule

def combine(tag=2017,threshold=0.16,alpha=0):
    print("Stage 3")
    print("Multiprocessing")
    up, lp, valueset_list = joblib.load(f"materials/materials_{tag}_js.pkl")
    x_train,x_test,y_train,y_test = joblib.load(f"materials/boxoutdata_{tag}_100_js.pkl")
    if not os.path.exists("materials/compressed_%d_%d_reallocated_js.pkl"%(tag,threshold*100)):
        print("X_train's sum:",x_train.sum()) 
        print("Threshold:",threshold)
        pool = Pool(processes=60)
        res_object = [pool.apply_async(compress_values,args=(x_train[:,i],y_train,x_test[:,i],i,valueset_list[i],threshold,alpha)) for i in range(x_train.shape[1])]
        res_train = [r.get()[0] for r in res_object]
        res_test = [r.get()[1] for r in res_object]
        res_rules = [r.get()[2] for r in res_object]
        res_valueset = [r.get()[3] for r in res_object]
        pool.close()
        pool.join()
        x_train = np.concatenate(res_train,axis=1)
        x_test = np.concatenate(res_test,axis=1)
        #joblib.dump([x_train,x_test,y_train,y_test],"materials/compressed_%d_%d_combined_js.pkl"%(tag,threshold*100))
        #bundle
        x_train, bundle_rule = bundle(x_train,threshold)
        for i in bundle_rule:
            for rule in bundle_rule[i]:
                indicies = x_test[:,i]==rule[0]
                x_test[indicies,rule[1]] = rule[2] 
                if i != rule[1]:
                    x_test[:,i] = 0
        joblib.dump([x_train,x_test,y_train,y_test],"materials/compressed_%d_%d_reallocated_js.pkl"%(tag,threshold*100))
        joblib.dump([res_rules,res_valueset,bundle_rule],"materials/compressed_%d_%d_material_js.pkl"%(tag,threshold*100))#used for processing other samples
        print(x_train.sum()) 

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_utils.load_dataset('ember')
    preprocess(x_train, x_test, 2017)
    for threshold in [0.08]:#[0.01,0.04,0.08,0.16]:
        combine(2017,threshold,0)

