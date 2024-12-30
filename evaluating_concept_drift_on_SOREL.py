import os
import joblib
import torch
import numpy as np
import ember
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from core import data_utils
from core import model_utils
from core import utils
from core import constants
max_workers = cpu_count()
saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")

features, feature_names, name_feat, feat_name = data_utils.load_features(constants.features_to_exclude['ember'],'ember')

import lmdb
from torch.utils import data
import sqlite3
import json
import tqdm
import msgpack
import zlib
from models import *

def get_model_by_name(name):
    if name == 'basenn':
        r =  get_base_nn()
        return r[4],r[6]#model and processor
    if name == 'baselgbm':
        r = get_base_lightgbm()
        return r[4],r[6]#model and processor
    if name == 'ltnn':
        r = get_ltnn()
        return r[4],r[6]#model and processor
    if name == 'binarized_model':
        r = get_binarized_model()
        return r[4],r[6]#model and processor
    if name == 'histogram_model':
        r = get_compressed_model('histogram','2017','16')
        return r[4],r[6]#model and processor
    if name == 'compressed_model':
        r = get_compressed_model(False,'2017','16')
        return r[4],r[6]#model and processor
    if name == 'compressed_model_True':
        r = get_compressed_model(True,'2017','8')
        return r[4],r[6]#model and processor
    if name == "pad" or "pad" in name:
        r = get_pad_model(name)
        return r[4],r[6]#model and processor
    if name == 'compressed_model_bundle':
        r = get_compressed_model('bundle','2017','8')
        return r[4],r[6]#model and processor
    if name == 'outlier_model':
        r = model_utils.load_model('nn','ember',saved_dir,"outlier_nn")
        return r,'outlier'#model and processor
    if name == 'combined_model':
        r = model_utils.load_model('nn','ember',saved_dir,"combined_nn")
        return r,'combined'#model and processor
    if ('density' in name or 'crop' in name) and 'bundle' in name:
        r = get_compressed_density_model('bundle','2017','8',name.split('_')[-1])#2 for 16
        return r[4],r[6]
    if ('density' in name or 'crop' in name) and 'bundle' not in name:
        r = get_compressed_density_model(True,'2017','8',name.split('_')[-1])#2 for 16
        return r[4],r[6]#model and processor
    else:
        raise Exception("Wrong model name!")

def features_postproc_func(x):
    x = np.asarray(x[0], dtype=np.float32)
    lz = x < 0
    gz = x > 0
    x[lz] = - np.log(1 - x[lz])
    x[gz] = np.log(1 + x[gz])
    return x

class LMDBReader(object):
    def __init__(self, path, postproc_func=None):
        self.env = lmdb.open(path, readonly=True, map_size=1e13, max_readers=1024)
        self.postproc_func = postproc_func

    def __call__(self, key):
        with self.env.begin() as txn:
            x = txn.get(key.encode('ascii'))
        if x is None:return None
        x = msgpack.loads(zlib.decompress(x),strict_map_key=False)
        x = np.asarray(x[0], dtype=np.float32)
        if self.postproc_func is not None:
            x = self.postproc_func(x)
        return x

validation_test_split =  1547279640.0
# This is the timestamp that splits training data from validation data
train_validation_split = 1543542570.0
concept_drift = 1514736000

class Dataset(data.Dataset):
    tags = ["adware", "flooder", "ransomware", "dropper", "spyware", "packed",
            "crypto_miner", "file_infector", "installer", "worm", "downloader"]

    def __init__(self, metadb_path, features_lmdb_path,
                 return_malicious=True, return_counts=True, return_tags=True, return_shas=False,
                 mode='train', binarize_tag_labels=True, n_samples=None, remove_missing_features=True,
                 postprocess_function=features_postproc_func):

        self.return_counts = return_counts
        self.return_tags = return_tags
        self.return_malicious = return_malicious
        self.return_shas = return_shas
        self.mode = mode
        
        self.features_lmdb_reader = LMDBReader(features_lmdb_path, postproc_func=postprocess_function)


        retrieve = ["sha256"]
        if return_malicious:
            retrieve += ["is_malware"]
        if return_counts:
            retrieve += ["rl_ls_const_positives"]
        if mode == 'concept':
            retrieve += ["rl_fs_t"]
        if return_tags:
            retrieve.extend(Dataset.tags)
        conn = sqlite3.connect(metadb_path)
        cur = conn.cursor()
        query = 'select ' + ','.join(retrieve)
        query += " from meta"

        if mode == 'train':
            query += ' where(rl_fs_t <= {})'.format(train_validation_split)
        elif mode == 'validation':
            query += ' where((rl_fs_t >= {}) and (rl_fs_t < {}))'.format(train_validation_split,
                                                                         validation_test_split)
        elif mode == 'test':
            query += ' where(rl_fs_t >= {})'.format(validation_test_split)
        elif mode == 'concept':
            query += ' where(rl_fs_t >= {})'.format(concept_drift)
        else:
            raise ValueError('invalid mode: {}'.format(mode))

        print('Opening Dataset at {} in {} mode.'.format(metadb_path, mode))

        if type(n_samples) != type(None):
            query += ' limit {}'.format(n_samples)
        vals = cur.execute(query).fetchall()
        conn.close()
        print(f"{len(vals)} samples loaded.")
        retrieve_ind = dict(zip(retrieve, list(range(len(retrieve)))))
        # map the items we're retrieving to an index
        if remove_missing_features=='scan':
            print("Removing samples with missing features...")
            indexes_to_remove = []
            print("Checking dataset for keys with missing features.")
            temp_env = lmdb.open(features_lmdb_path, readonly=True, map_size=1e13, max_readers=256)
            with temp_env.begin() as txn:
                for index, item in tqdm.tqdm(enumerate(vals), total=len(vals), mininterval=.5, smoothing=0.):
                    if txn.get(item[retrieve_ind['sha256']].encode('ascii')) is None:
                        indexes_to_remove.append(index)
            indexes_to_remove = set(indexes_to_remove)
            vals = [value for index, value in enumerate(vals) if index not in indexes_to_remove]
            print(f"{len(indexes_to_remove)} samples had no associated feature and were removed.")
            print(f"Dataset now has {len(vals)} samples.")
        elif (remove_missing_features is False) or (remove_missing_features is None):
            pass
        else:
            # assume filepath
            print(f"Trying to load shas to ignore from {remove_missing_features}...")
            with open(remove_missing_features, 'r') as f:
                shas_to_remove = json.load(f)
            shas_to_remove = set(shas_to_remove)
            vals = [value for value in vals if value[retrieve_ind['sha256']] not in shas_to_remove]
            print(f"Dataset now has {len(vals)} samples.")
        self.keylist = list(map(lambda x: x[retrieve_ind['sha256']], vals))
        self.times = list(map(lambda x: x[retrieve_ind['rl_fs_t']], vals))
        if self.return_malicious:
            self.labels = list(map(lambda x: x[retrieve_ind['is_malware']], vals))
        if self.return_counts:
            self.count_labels = list(map(lambda x: x[retrieve_ind['rl_ls_const_positives']], vals))
        if self.return_tags:
            self.tag_labels = np.asarray([list(map(lambda x: x[retrieve_ind[t]], vals)) for t in Dataset.tags]).T
            if binarize_tag_labels:
                self.tag_labels = (self.tag_labels != 0).astype(int)
                
    def __len__(self):
        return len(self.keylist)

    def __getitem__(self, index):
        labels = {}
        key = self.keylist[index]
        features = self.features_lmdb_reader(key)
        if self.return_malicious:
            labels['malware'] = self.labels[index]
        if self.return_counts:
            labels['count'] = self.count_labels[index]
        if self.return_tags:
            labels['tags'] = self.tag_labels[index]
        if self.mode == 'concept':
            labels['times'] = self.times[index]
        if self.return_shas:
            return key, features, labels
        else:
            return features, labels

def evaluate_aut(y_test,y_pred,times):
    timelines = np.array([
                 1514736000,1517414400,1519833600,1522512000,1525104000,1527782400,
                 1530374400,1533052800,1535731200,1538323200,1541001600,1543593600,
                 1546272000,1548950400,1551369600,1554048000,1556640000])
    f1_list = []
    goodware_count = []
    malware_count = []
    for i in range(len(timelines)-1):
        indicies = (times>=timelines[i])&(times<timelines[i+1])
        y_t,y_p = y_test[indicies], y_pred[indicies]
        f1_list.append(f1_score(y_t,y_p))
    return f1_list, utils.aut_score(f1_list)


def evaluation(model_name):
    model, processor = get_model_by_name(model_name)
    ds = Dataset(metadb_path=constants.SOREL_META,#"../meta.db",
                 features_lmdb_path=constants.SOREL_EMBERFEAT,#"ember_features/",
                 return_malicious=True,
                 return_counts=False,
                 return_tags=False,
                 return_shas=False, mode='concept',
                 remove_missing_features=constants.SOREL_MISSING,#'SOREL-20M/shas_missing_ember_features.json',#'scan',
                 postprocess_function=None)
    params = {'batch_size': 8192,
            'shuffle': False,
            'num_workers': 60}
    
    generator = data.DataLoader(ds, **params)
    y_true = []
    y_pred = []
    times = []
    if 'outlier' in model_name:#ablation study with only outlier procession
        _, processor = get_model_by_name('compressed_model_bundle')
        processor.rules = []
    if 'combined' in model_name:#ablation study without combination
        _, processor = get_model_by_name('compressed_model_bundle')
        processor.bundle_rule = []
    for i, (features, labels) in tqdm.tqdm(enumerate(generator)):
        if 'compressed' in model_name or 'pad' in model_name or 'histogram' in model_name or 'outlier' in model_name or 'combined' in model_name:
            features = processor.process(features.numpy())
        if 'binarized' in model_name:
            features[features!=0] = 1
        y_true.append(np.array(labels['malware']))
        times.append(labels['times'])
        if 'pad' not in model_name:
            r = model.predict(features) > 0.5
        else:
            y_cent, y_prob, y_true = model.model.inference(utils.data_iter(100, features, labels['malware'], False))
            indicator_flag = model.model.indicator(y_prob).cpu().numpy()
            r = y_cent[:,1] > 0.5
            #r[~indicator_flag] = True
        y_pred.append(np.array(r))
    #print(y_pred,y_true)
    y_true = np.concatenate(y_true).astype(int)
    y_pred = np.concatenate(y_pred).astype(int)
    times = np.concatenate(times)
    print(f"Malware: {y_true.sum()} Benign:{y_true.shape[0]-y_true.sum()}")
    f1_list, aut = evaluate_aut(y_true,y_pred,times)
    with open("materials/evaluation_on_sorel.txt",'a') as f:
        f.write(f"{model_name} aut: {aut} - {f1_list}\n")
        joblib.dump([y_true,y_pred,f1_list,times],f"materials/{model_name}_sorel_result.pkl")

if __name__ == '__main__':
    for model_name in ['basenn','baselgbm','ltnn','binarized_model','histogram_model','compressed_model','compressed_model_bundle','compressed_model_bundle_density0','basepad','padXX']:
        #for model_name in ['outlier_model','combined_model']:
        evaluation(model_name)
