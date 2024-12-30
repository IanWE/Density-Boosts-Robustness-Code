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

import random

# FEATURES
def load_features(feats_to_exclude, dataset='ember', selected=False, vrb=False):
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
        feature_names, non_hashed, hashed, feasible = load_drebin_features(feats_to_exclude, selected)

    else:
        raise NotImplementedError('Dataset {} not supported'.format(dataset))

    feature_ids = list(range(feature_names.shape[0]))
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
        loggger.debug('\nList of non-hashed features:')
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
    df = pd.read_csv('pdfs/data.csv')
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


def load_drebin_features(infeas, selected=False):
    """ Return the list of Drebin features.

    Due to the huge number of features we will use the vectorizer file saved
    during the preprocessing.

    :return:
    """
    prefixes = {
        'activitylist': 'manifest',
        'broadcastreceiverlist': 'manifest',
        'contentproviderlist': 'manifest',
        'servicelist': 'manifest',
        'intentfilterlist':'manifest',
        'requestedpermissionlist':'manifest',
        'hardwarecomponentslist':'manifest',
        'restrictedapilist':'code',
        'usedpermissionslist':'code',
        'suspiciousapilist':'code',
        'urldomainlist': 'code'
    }
    if selected==True:
        feat_file = os.path.join(constants.SAVE_FILES_DIR,"selected_features.pkl")
    else:
        feat_file = os.path.join(constants.SAVE_FILES_DIR,"original_features.pkl")
    # Check if the feature file is available, otherwise create it
    if not os.path.isfile(feat_file):
        load_drebin_dataset(selected=selected)
    feature_names = joblib.load(feat_file)
    n_f = feature_names.shape[0]

    feasible = [i for i in range(n_f) if feature_names[i].split('_')[0] not in infeas]
    hashed = [i for i in range(n_f) if prefixes[feature_names[i].split('_')[0]] == 'code']
    non_hashed = [i for i in range(n_f) if prefixes[feature_names[i].split('_')[0]] == 'manifest']

    return np.array(feature_names), non_hashed, hashed, feasible


# DATA SETS
def load_dataset(dataset='ember', selected=True, processor=None):
    if dataset == 'ember':
        x_train, y_train, x_test, y_test = load_ember_dataset()

    elif dataset == 'pdf':
        x_train, y_train, x_test, y_test = load_pdf_dataset()
    elif dataset == 'drebin':
        x_train, y_train, x_test, y_test = load_drebin_dataset(selected)
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
    def __init__(self,up,lp,valueset_list,rules,c_valueset_list,threshold,new=False,bundle_rule=[]):
        self.up = up
        self.lp = lp
        self.threshold = threshold
        self.valueset_list = valueset_list
        self.rules = rules
        self.c_valueset_list = c_valueset_list
        self.new = new
        self.bundle_rule = bundle_rule

    def process(self,x):
        for i in range(x.shape[1]):
            x_i = x[:,i]
            if self.lp[i]==self.up[i] and self.new:
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

def load_compressed_ember(tag, ratio=8, binarization=False):
    if binarization == False:
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_reallocated.pkl"))
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}.pkl"))
        rules,c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,ratio,binarization)
    elif binarization == True:
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_reallocated_js.pkl"))
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}_js.pkl"))
        rules,c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_js.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,ratio,binarization)
    elif binarization == 'bundle':
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_reallocated_js.pkl"))
        up,lp,valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"materials_{tag}_js.pkl"))
        rules,c_valueset_list,bundle_rule = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_js.pkl"))
        processor = Processor(up,lp,valueset_list,rules,c_valueset_list,ratio,binarization,bundle_rule)
    elif binarization == 'histogram':
        x_train, x_test, y_train, y_test = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_histogram.pkl"))
        c_valueset_list = joblib.load(os.path.join(constants.SAVE_FILES_DIR,f"compressed_{tag}_{ratio}_material_histogram.pkl"))
        processor = HistogramProcessor(c_valueset_list,ratio)
    else:
        print("Wrong compression method!")
        return 
    return x_train, y_train, x_test, y_test, processor

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


def load_drebin_dataset(selected=False):
    """ Vectorize and load the Drebin dataset.

    :param selected: (bool) if true return feature subset selected with Lasso
    :return
    """
    AllMalSamples = os.listdir(constants.DREBIN_DATA_DIR+'malware/')
    AllMalSamples = list(map(lambda x:constants.DREBIN_DATA_DIR+'malware/'+x,AllMalSamples))
    AllGoodSamples = os.listdir(constants.DREBIN_DATA_DIR+'benign/')
    AllGoodSamples = list(map(lambda x:constants.DREBIN_DATA_DIR+'benign/'+x,AllGoodSamples))
    AllSampleNames = AllMalSamples + AllGoodSamples
    AllSampleNames = list(filter(lambda x:x[-5:]==".data",AllSampleNames))#extracted features

    # label malware as 1 and goodware as -1
    Mal_labels = np.ones(len(AllMalSamples))
    Good_labels = np.empty(len(AllGoodSamples))
    Good_labels.fill(-1)
    y = np.concatenate((Mal_labels, Good_labels), axis=0)
    logger.info("Label array - generated")

    # First check if the processed files are already available,
    # load them directly if available.
    if os.path.isfile(constants.SAVE_FILES_DIR+"drebin.pkl"):
        x_train,y_train,x_test,y_test,features,x_train_names,x_test_names = joblib.load(constants.SAVE_FILES_DIR+"drebin.pkl")
    else:
        FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                                           binary=True)
        x_train_samplenames, x_test_samplenames, y_train, y_test = train_test_split(AllSampleNames, y, test_size=0.2,random_state=0)
        x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
        x_test = FeatureVectorizer.transform(x_test_samplenames)
        features = FeatureVectorizer.get_feature_names_out()
        joblib.dump([x_train,y_train,x_test,y_test,features,x_train_samplenames,x_test_samplenames],constants.SAVE_FILES_DIR+"drebin.pkl")
        joblib.dump(np.array(features),os.path.join(constants.SAVE_FILES_DIR,"original_features.pkl"))
    if selected:
        selector_path = os.path.join(constants.SAVE_FILES_DIR,"selector.pkl")
        if os.path.isfile(selector_path):
            selector = joblib.load(selector_path)
        else:
            random.seed(1)
            lsvc = LinearSVC(C=0.05, penalty="l1", dual=False).fit(x_train, y_train)
            r=lsvc.predict(x_train)
            logger.info("Acc of selector:",(r==y_train).sum()/y_train.shape[0])
            selector = SelectFromModel(lsvc, prefit=True)
            logger.debug("Features after selection: %d",np.where(selector.get_support()==True)[0].shape[0])
            joblib.dump(selector,selector_path)
        x_train = selector.transform(x_train) 
        x_test = selector.transform(x_test) 
        joblib.dump(np.array(features)[selector.get_support()],os.path.join(constants.SAVE_FILES_DIR,"selected_features.pkl"))
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

