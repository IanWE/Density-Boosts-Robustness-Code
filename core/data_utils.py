import os

import ember
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer as TF
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

from core import ember_feature_utils, constants
from logger import logger
from multiprocessing import Pool

from tqdm import tqdm
import json
import random

# FEATURES
def load_features(feats_to_exclude, dataset='ember', year='2015', selected=False, dense_features=[], vrb=False):
    """ Load the features and exclude those in list.
    :param feats_to_exclude: (list) list of features to exclude
    :param dataset: (str) name of the dataset being used
    :param selected: (bool) if true load only Lasso selected features for Drebin
    :param vrb: (bool) if true print debug strings

    :return: (dict, array, dict, dict) feature dictionaries
    """

    if dataset == 'ember':
        feature_names = np.array(ember_feature_utils.build_feature_names())
        non_hashed = ember_feature_utils.get_non_hashed_features()
        hashed = ember_feature_utils.get_hashed_features()

    elif dataset == 'pdf':
        feature_names, non_hashed, hashed = load_pdf_features()

    elif dataset == 'drebin':
        feature_names, non_hashed, hashed, feasible = load_drebin_features(feats_to_exclude, year, selected, dense_features)

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    feature_ids = list(range(len(feature_names)))
    # The `features` dictionary will contain only numerical IDs
    features = {
        'all': feature_ids,
        'non_hashed': non_hashed,
        'hashed': hashed
    }
    name_feat = dict(zip(feature_names, feature_ids))
    feat_name = dict(zip(feature_ids, feature_names))

    if dataset != 'drebin':
        feasible = features['non_hashed'].copy()
        for u_f in feats_to_exclude:
            feasible.remove(name_feat[u_f])
    features['feasible'] = feasible

    if vrb:
        logger.debug(
            'Total number of features: {}\n'
            'Number of non hashed features: {}\n'
            'Number of hashed features: {}\n'
            'Number of feasible features: {}\n'.format(
                len(features['all']),
                len(features['non_hashed']),
                len(features['hashed']),
                len(features['feasible'])
            )
        )
        logger.debug('\nList of non-hashed features:')
        logger.debug(
            ['{}: {}'.format(f, feat_name[f]) for f in features['non_hashed']]
        )
        logger.debug('\nList of feasible features:')
        logger.debug(
            ['{}: {}'.format(f, feat_name[f]) for f in features['feasible']]
        )

    return features, feature_names, name_feat, feat_name


def load_pdf_features():
    """ Load the PDF dataset feature list

    :return: (ndarray) array of feature names for the pdf dataset
    """

    arbitrary_feat = [
        'author_dot',
        'keywords_dot',
        'subject_dot',
        'author_lc',
        'keywords_lc',
        'subject_lc',
        'author_num',
        'keywords_num',
        'subject_num',
        'author_oth',
        'keywords_oth',
        'subject_oth',
        'author_uc',
        'keywords_uc',
        'subject_uc',
        'createdate_ts',
        'moddate_ts',
        'title_dot',
        'createdate_tz',
        'moddate_tz',
        'title_lc',
        'creator_dot',
        'producer_dot',
        'title_num',
        'creator_lc',
        'producer_lc',
        'title_oth',
        'creator_num',
        'producer_num',
        'title_uc',
        'creator_oth',
        'producer_oth',
        'version',
        'creator_uc',
        'producer_uc'
    ]
    #feature_names = np.load('df_features.npy')
    df = pd.read_csv(os.path.join(constants.PDF_DATA_DIR,constants.PDF_DATA_CSV))
    feature_names = df.columns.values[2:]
    non_hashed = [np.searchsorted(feature_names, f) for f in sorted(arbitrary_feat)]

    hashed = list(range(feature_names.shape[0]))
    hashed = list(set(hashed) - set(non_hashed))

    return feature_names, non_hashed, hashed


def build_feature_names(dataset='ember'):
    """ Return the list of feature names for the specified dataset.

    :param dataset: (str) dataset identifier
    :return: (list) list of feature names
    """

    features, feature_names, name_feat, feat_name = load_features(
        feats_to_exclude=[],
        dataset=dataset
    )

    return feature_names.tolist()


def load_drebin_features(infeas, year='2015', selected=False, dense_features=[]):
    """ Return the list of Drebin features.

    Due to the huge number of features we will use the vectorizer file saved
    during the preprocessing.

    :return:
    """
    #prefixes = {
    #    'activitylist': 'manifest',
    #    'broadcastreceiverlist': 'manifest',
    #    'contentproviderlist': 'manifest',
    #    'servicelist': 'manifest',
    #    'intentfilterlist':'manifest',
    #    'requestedpermissionlist':'manifest',
    #    'hardwarecomponentslist':'manifest',
    #    'restrictedapilist':'code',
    #    'usedpermissionslist':'code',
    #    'suspiciousapilist':'code',
    #    'urldomainlist': 'code'
    #}
    prefixes = {
        '':'manifest',
        'activities': 'manifest',
        's_and_r': 'manifest',
        'providers': 'manifest',
        'intents':'manifest',
        'app_permissions':'manifest',
        'api_permissions':'code',
        'interesting_calls':'code',#dangerous calls
        'api_calls':'code',
        'urls': 'code'
    }
    if selected==True:
        feat_file = os.path.join(constants.DREBIN_DATA_DIR,f"selected_features_{year}.pkl")
    else:
        feat_file = os.path.join(constants.DREBIN_DATA_DIR,f"original_features_{year}.pkl")
    # Check if the feature file is available, otherwise create it
    if not os.path.isfile(feat_file):
        load_drebin_dataset(year, selected=selected)
    feature_names = joblib.load(feat_file)
    if len(dense_features) != 0:
        feature_names = [feature_names[i] for i in dense_features]
    n_f = len(feature_names)

    feasible = [i for i in range(n_f) if feature_names[i].split('::')[0] not in infeas]
    hashed = [i for i in range(n_f) if prefixes[feature_names[i].split('::')[0]] == 'code']
    non_hashed = [i for i in range(n_f) if prefixes[feature_names[i].split('::')[0]] == 'manifest']

    return feature_names, non_hashed, hashed, feasible


# DATA SETS
def load_dataset(dataset='ember', selected=True, processor=None, year=2015):
    """Load a specified dataset based on the dataset name.
    @param dataset: (str) The name of the dataset to load, default is 'ember'.
    @param selected: (bool) A flag for dataset selection, default is True.
    @param processor: (object) A processor used to process data, default is None.
    @param year: (int) The year of the dataset, only for drebin

    return: (numpy.ndarray) x_train, (numpy.ndarray) y_train, (numpy.ndarray) x_test, (numpy.ndarray) y_test
    """
    if dataset == 'ember':
        x_train, y_train, x_test, y_test = load_ember_dataset()
    elif dataset == 'pdf':
        x_train, y_train, x_test, y_test = load_pdf_dataset()
    elif dataset == 'drebin':
        x_train, y_train, x_test, y_test = load_drebin_dataset(year,selected)
    elif dataset == "ember2018":
        x_train, y_train, x_test, y_test = load_ember_2018()
    elif dataset == "emberall":
        x_train, y_train, x_test, y_test = data_utils.load_dataset('ember')
        x_train_2018, y_train_2018, x_test_2018, y_test_2018 = data_utils.load_dataset('ember2018')
        x_train = np.concatenate([x_train,x_test,x_train_2018,x_test_2018],axis=0)
        y_train = np.concatenate([y_train,y_test,y_train_2018,y_test_2018],axis=0)
        emberdf = ember.read_metadata("datasets/ember_2017_2/")
        emberdf2018 = ember.read_metadata("datasets/ember2018/")
        emberdf = pd.concat([emberdf,emberdf2018])
        emberdf = emberdf[emberdf['label']!=-1] #1600000
        #x_test = x_test_2018
        #y_test = y_test_2018
        return x_train, y_train, emberdf
    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    return x_train, y_train, x_test, y_test

class Processor(object):
    def __init__(self,up,lp,valueset_list,rules,c_valueset_list,binarization=False,bundle_rule=[]):
        self.up = up
        self.lp = lp
        #self.threshold = threshold
        self.valueset_list = valueset_list
        self.rules = rules
        self.c_valueset_list = c_valueset_list
        self.binarization = binarization
        self.bundle_rule = bundle_rule

    def process(self,x):
        for i in range(x.shape[1]):
            x_i = x[:,i]
            if self.lp[i]==self.up[i] and self.binarization:
                indices = self.lp[i]==x_i
                x_i[indices] = 0
                x_i[~indices] = 1
            else:
                x_i[x_i<self.lp[i]] = self.lp[i]
                x_i[x_i>self.up[i]] = self.up[i]
        if not self.rules:
            return x
        x_copy = x.copy()
        for i in range(0,x.shape[1]):
            #evenly cut it into 100 sections
            x_i = x[:,i]
            x_clip = x_copy[:,i]
            valueset = self.valueset_list[i]
            for vi in range(len(valueset)-1): #redundant features are eliminated here
                x_i[(x_clip>=valueset[vi])&(x_clip<valueset[vi+1])]=valueset[vi]
            x_i[x_clip<valueset[0]] = valueset[0]
            c_valueset = self.c_valueset_list[i]
            rule = self.rules[i]
            if rule is not None:
                x_i[x_i<c_valueset[0]] = c_valueset[0]
                x_i[x_i>c_valueset[-1]] = c_valueset[-1]
                for v in rule:
                    for r in rule[v]:
                        x_i[x_i==r] = v
                #move outside 
                gap = 1/len(c_valueset)
                x_orig = x_i.copy()
                for idx,j in enumerate(c_valueset):
                    x_i[x_orig==j] = idx*gap
        if not self.bundle_rule:
            return x
        for i in self.bundle_rule:
            #traveling all rules
            for rule in self.bundle_rule[i]:
                indicies = x[:,i] == rule[0]
                x[indicies,rule[1]] = rule[2]
                #if it is a different feature, remove the origin feature
                if i != rule[1]:
                    x[:,i] = 0
        return x

class HistogramProcessor:
    def __init__(self,valuesets,threshold):
        self.threshold = threshold
        self.valuesets = valuesets
    def process(self,x):
        for i in range(0,x.shape[1]):
            valueset = self.valuesets[i]
            x_i = x[:,i]
            x_i[x_i<valueset[0]] = valueset[0]
            x_ = x_i.copy()
            gap = 1/len(valueset)
            for i in range(len(valueset)-1):
                x_i[(x_>=valueset[i])&(x_<valueset[i+1])] = gap * i
        return x

class DrebinProcessor:
    def __init__(self,selected_features,bundle_rule,f_in):
        self.selected_features = selected_features
        self.bundle_rule = bundle_rule
        self.f_in = f_in
    def process(self,x):
        if len(x.shape)!=2:
            raise NotImplementedError('input dim must be 2!')
        if hasattr(x,'todense'):            
            x = np.array(x[:,self.selected_features].todense())
        else:
            x = x[:,self.selected_features]
        for i in self.bundle_rule:
            for rule in self.bundle_rule[i]:
                indicies = x[:,i]==rule[0]
                x[indicies,rule[1]] = rule[2]
                if i != rule[1]:
                    x[:,i] = 0
        x = x[:,self.f_in]
        return x

def load_compressed_pdf(tag='pdf', ratio=8, binarization='bundle'):
    """Load compressed PDF data based on specified parameters.
    @param tag: (str) Tag for the data, used for saving and loading dataset.
    @param ratio: (int) Compression ratio, default is 8.
    @param binarization: (str or bool) Binarization method, default is 'bundle'.

    return: (numpy.ndarray) x_train, (numpy.ndarray) y_train, (numpy.ndarray) x_test, (numpy.ndarray) y_test, (object) processor
    """
    if binarization == 'bundle':
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_reallocated.pkl"))
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}.pkl"))
        rules,c_valueset_list,bundle_rule = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization,bundle_rule)
    elif binarization == False:
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_reallocated_sc.pkl"))
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}_sc.pkl"))
        rules,c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_sc.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization,[])
    elif binarization == 'histogram':
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_histogram.pkl"))
        c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_histogram.pkl"))
        processor = HistogramProcessor(c_valueset_list,ratio)
    else:
        print("Wrong compression method!")
        return 
    return x_train, y_train, x_test, y_test, processor


#please run process.py first boxoutdata_2017_100_new.pkl
def load_compressed_ember(tag, ratio=8, binarization=False):
    if binarization == False:
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_reallocated.pkl"))
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}.pkl"))
        rules,c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization)
    elif binarization == True:
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_reallocated_js.pkl"))
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}_js.pkl"))
        rules,c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_js.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization)
    elif binarization == 'bundle':
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_reallocated_js.pkl"))
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}_js.pkl"))
        rules,c_valueset_list,bundle_rule = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_js.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization,bundle_rule)
    elif binarization == 'histogram':
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_histogram.pkl"))
        c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_histogram.pkl"))
        processor = HistogramProcessor(c_valueset_list,ratio)
    else:
        print("Wrong compression method!")
        return 
    return x_train, y_train, x_test, y_test, processor

def load_compressed_drebin(tag='2015', ratio=16):
    x_train,x_test,y_train,y_test = joblib.load(f"materials/compressed_drebin_{tag}_{ratio}.pkl")
    selected_features,bundle_rule,f_in = joblib.load(f"materials/compressed_drebin_{tag}_{ratio}_material.pkl")#used for processing other samples
    processor = DrebinProcessor(selected_features,bundle_rule,f_in)
    return x_train, x_test, y_train, y_test, processor

def load_processor(tag,ratio,binarization):
    """Load the processor
    :param tag: (str) tag of the processor
    :param ratio: (float) compression rate
    :param binarization: (bool) whether to use the binarization mechanism
    """
    if binarization == False:
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}.pkl"))
        rules,c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization)
    elif binarization == True:
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}_js.pkl"))
        rules,c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_js.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization)
    elif binarization == 'histogram':
        c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_histogram.pkl"))
        processor = HistogramProcessor(c_valueset_list,ratio)
    elif binarization == 'bundle':
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}_js.pkl"))
        rules,c_valueset_list,bundle_rule = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_js.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,binarization,bundle_rule)
    else:
        print("Wrong compression method!")
    return processor


def load_ember_2018():
    """ Return train and test data from EMBER.

    :return: (array, array, array, array)
    """

    # Perform feature vectorization only if necessary.
    x_train, y_train, x_test, y_test = ember.read_vectorized_features(
         constants.EMBER_DATA_DIR_2018,
         feature_version=2
    )
    x_train = x_train.astype(dtype='float64')
    x_test = x_test.astype(dtype='float64')

    # Get rid of unknown labels
    x_train = x_train[y_train != -1]
    y_train = y_train[y_train != -1]
    x_test = x_test[y_test != -1]
    y_test = y_test[y_test != -1]

    return x_train, y_train, x_test, y_test
    


# noinspection PyBroadException
def load_ember_dataset():
    """ Return train and test data from EMBER.

    :return: (array, array, array, array)
    """

    # Perform feature vectorization only if necessary.
    try:
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            constants.EMBER_DATA_DIR,
            feature_version=2
        )

    except:
        ember.create_vectorized_features(
            constants.EMBER_DATA_DIR,
            feature_version=2
        )
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            constants.EMBER_DATA_DIR,
            feature_version=2
        )

    x_train = x_train.astype(dtype='float64')
    x_test = x_test.astype(dtype='float64')

    # Get rid of unknown labels
    x_train = x_train[y_train != -1]
    y_train = y_train[y_train != -1]
    x_test = x_test[y_test != -1]
    y_test = y_test[y_test != -1]

    return x_train, y_train, x_test, y_test


def load_pdf_dataset():
    import pandas as pd
    df = pd.read_csv(os.path.join(constants.PDF_DATA_DIR,constants.PDF_DATA_CSV))

    mwdf = df[df['class']==True]
    train_mw, test_mw = train_test_split(mwdf, test_size=0.4, random_state=42)

    gwdf = df[df['class']==False]
    train_gw, test_gw = train_test_split(gwdf, test_size=0.4, random_state=42)

    # Merge dataframes
    train_df = pd.concat([train_mw, train_gw])
    test_df = pd.concat([test_mw, test_gw])

    # Transform to numpy
    y_train = train_df['class'].apply(lambda x:1 if x else 0).to_numpy()
    y_test = test_df['class'].apply(lambda x:1 if x else 0).to_numpy()

    x_train_filename = train_df['filename'].to_numpy()
    x_test_filename = test_df['filename'].to_numpy()

    x_train = train_df.drop(columns=['class', 'filename']).to_numpy()
    x_test = test_df.drop(columns=['class', 'filename']).to_numpy()
    x_train = x_train.astype(dtype='float64')
    x_test = x_test.astype(dtype='float64')

    # Save the file names corresponding to each vector into separate files to
    # be loaded during the attack
    np.save(os.path.join(constants.SAVE_FILES_DIR, 'x_train_filename'), x_train_filename)
    np.save(os.path.join(constants.SAVE_FILES_DIR, 'x_test_filename'), x_test_filename)

    return x_train, y_train, x_test, y_test


def load_pdf_train_test_file_names():
    """ Utility to return the train and test set file names for PDF data

    :return: (ndarray, ndarray)
    """
    train_files_npy = os.path.join(constants.SAVE_FILES_DIR, 'x_train_filename.npy')
    train_files = np.load(train_files_npy, allow_pickle=True)

    test_files_npy = os.path.join(constants.SAVE_FILES_DIR, 'x_test_filename.npy')
    test_files = np.load(test_files_npy, allow_pickle=True)

    return train_files, test_files


def _vectorize(x, y):
    vectorizer = DictVectorizer()
    x = vectorizer.fit_transform(x)
    y = np.asarray(y)
    return x, y, vectorizer

def get_sample_path_drebin(year):
    samples_info = pd.read_csv(os.path.join(constants.DREBIN_DATA_DIR,"samples_info.csv"))
    train_samplenames = samples_info[((samples_info.vt_detection>=10)|(samples_info.vt_detection==0))&(samples_info.dex_date<year)].sha256.tolist()
    test_samplenames = samples_info[((samples_info.vt_detection>=10)|(samples_info.vt_detection==0))&(samples_info.dex_date>year)].sha256.tolist()
    return train_samplenames,test_samplenames

def load_drebin_dataset(year='2015',selected=False, end='2019'):
    """ Vectorize and load the Drebin dataset.

    :param selected: (bool) if true return feature subset selected with Lasso
    :return
    """
    #samples_info = pd.read_csv('../samples_info.csv')
    #extracted_features = joblib.load("datasets/drebin/raw_datasets.pkl")

    logger.info("Label array - generated")
    # First check if the processed files are already available,
    # load them directly if available.
    if os.path.isfile(constants.DREBIN_DATA_DIR+f"drebin_{year}_{end}.pkl"):
        x_train,y_train,x_test,y_test = joblib.load(constants.DREBIN_DATA_DIR+f"drebin_{year}_{end}.pkl")
        features = joblib.load(constants.DREBIN_DATA_DIR+f'original_features_{year}.pkl')
    else:
        raw_datasets_path = os.path.join(constants.DREBIN_DATA_DIR,"raw_datasets.pkl")
        y = np.array(json.load(open(os.path.join(constants.DREBIN_DATA_DIR,"extended-features-y.json"))))
        if os.path.isfile(raw_datasets_path):
            extracted_features = joblib.load(raw_datasets_path)
            samples_info = pd.read_json(os.path.join(constants.DREBIN_DATA_DIR,"extended-features-meta.json"))
        else:
            samples_info = pd.read_json(os.path.join(constants.DREBIN_DATA_DIR,"extended-features-meta.json"))
            meta = json.load(open(os.path.join(constants.DREBIN_DATA_DIR,"extended-features-X.json")))
            extracted_features = []
            for f in meta:
                d = list(f.keys())
                extracted_features.append("|==========|".join(d))
            joblib.dump(extracted_features,os.path.join(constants.DREBIN_DATA_DIR,"raw_datasets.pkl"))
        x_train_raw = [extracted_features[i] for i in samples_info[(samples_info.dex_date<year)].index.tolist()]
        y_train = np.array(y)[(samples_info.dex_date<year).tolist()]
        x_test_raw = [extracted_features[i] for i in samples_info[(samples_info.dex_date>year)&(samples_info.dex_date<end)].index.tolist()]
        y_test = np.array(y)[((samples_info.dex_date>year)&(samples_info.dex_date<end)).tolist()]
        cv = CountVectorizer(binary=True,tokenizer=lambda x:x.split('|==========|'))
        x_train = cv.fit_transform(x_train_raw)
        x_test = cv.transform(x_test_raw)
        joblib.dump(cv.get_feature_names_out(),os.path.join(constants.DREBIN_DATA_DIR,f'original_features_{year}.pkl'))
        joblib.dump([x_train,y_train,x_test,y_test],os.path.join(constants.DREBIN_DATA_DIR,f'drebin_{year}_{end}.pkl'))
    if selected:
        selector_path = os.path.join(constants.DREBIN_DATA_DIR,f"selector_{year}_{end}.pkl")
        if os.path.isfile(selector_path):
            selector = joblib.load(selector_path)
        else:
            random.seed(1)
            lsvc = LinearSVC(C=1, penalty="l2", dual=False).fit(x_train, y_train)
            r=lsvc.predict(x_train)
            logger.info("Acc of selector:",(r==y_train).sum()/y_train.shape[0])
            selector = SelectFromModel(lsvc, prefit=True, max_features=2000)
            #selector.fit(x_train,y_train)
            logger.debug("Features after selection: %d",np.where(selector.get_support()==True)[0].shape[0])
            joblib.dump(selector,selector_path)
        x_train = selector.transform(x_train) 
        x_test = selector.transform(x_test) 
        selected_features = [features[i] for i in np.where(selector.get_support()==True)[0]]
        joblib.dump(selected_features,os.path.join(constants.SAVE_FILES_DIR,"selected_features_{year}_{end}.pkl"))
    return x_train, y_train, x_test, y_test

def extract_features(bytez):
    #%%capture
    import sys
    import io
    origin_output = sys.stdout
    origin_err = sys.stderr
    fake_stdout = io.StringIO()
    sys.stdout = fake_stdout
    sys.stderr = fake_stdout
    #with open(filename,"rb") as f:
    #    bytez = f.read()
    extractor = ember.PEFeatureExtractor(2)
    f = extractor.feature_vector(bytez)
    #sys.stdout = origin_output
    #sys.stderr = origin_err
    return f

