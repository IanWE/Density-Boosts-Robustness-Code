# This the directory that contains the ember data set and model
#   ember_model_2017.txt
#   test_features.jsonl
#   train_features_0.jsonl
#   train_features_1.jsonl
#   train_features_2.jsonl
#   train_features_3.jsonl
#   train_features_4.jsonl
#   train_features_5.jsonl
#
# Change this to where you unzipped
# https://pubdata.endgame.com/ember/ember_dataset.tar.bz
EMBER_DATA_DIR = 'datasets/ember_2017_2/'
EMBER_DATA_DIR_2018 = 'datasets/ember2018/'
EMBER_MALWARE_DIR = 'datasets/malware-PE/'

# This is the directory containing the pdf dataset
PDF_DATA_DIR = 'datasets/PDFs/'
PDF_DATA_CSV = 'mimicus/data.csv' #An csv of extracted PDFs based on Mimicus

# Directory containing Drebin data
DREBIN_RAW_DIR = "../feature-extractor/results/"
DREBIN_DATA_DIR = "datasets/drebin/"

# Files for evaluation on SOREL dataset, which are all provided by SOREL
SOREL_META = "meta.db"
SOREL_EMBERFEAT = "ember_features/"
SOREL_MISSING = "SOREL-20M/shas_missing_ember_features.json"

# Path to local dir where to save trained models
SAVE_MODEL_DIR = 'models/'

# Path to directory where to save large files
SAVE_FILES_DIR = 'materials/'

# This path is used to store temporary pdf files needed by mimicus featureedit.py
TEMP_DIR = 'temp/'

# Controls whether some expensive assertions are done or not.
# When making changes to any logic in this file it can be helpful to turn this
# on. But when running experiments for real turning this off saves a fair
# amount of time.
DO_SANITY_CHECKS = False

VERBOSE = True

# Datasets sizes
NUM_SHAP_INTERACTIVITY_SAMPLES = 500
NUM_EMBER_FEATURES = 2381
NUM_PDF_FEATURES = 135
NUM_DREBIN_FEATURES = 2378905

EMBER_TRAIN_SIZE = 600000
EMBER_TRAIN_GW_SIZE = 300000
PDF_TRAIN_SIZE = 12053
PDF_TRAIN_GW_SIZE = 8036
DREBIN_TRAIN_SIZE = 86438

# ########################### #
# Genrerally useful constants #
# ########################### #
# Model identifiers
possible_model_targets = [
    'lightgbm',
    'nn',
    'rf',
    'linearsvm'
]

# Dataset identifies
possible_datasets = [
    'ember',
    'pdf',
    'drebin'
]

# Number of features per dataset
num_features = {
    'ember': NUM_EMBER_FEATURES,
    'pdf': NUM_PDF_FEATURES,
    'drebin': NUM_DREBIN_FEATURES
}

# Number of data points in training set
train_sizes = {
    'ember': EMBER_TRAIN_SIZE,
    'pdf': PDF_TRAIN_SIZE,
    'drebin': DREBIN_TRAIN_SIZE
}

# Feature malleability
possible_features_targets = {
    'all',
    'non_hashed',
    'feasible'
}

possible_knowledge = {
    'train',
    'test'
}

infeasible_features = [
    'avlength',
    'exports',
    'has_debug',
    'has_relocations',
    'has_resources',
    'has_signature',
    'has_tls',
    'imports',
    'major_subsystem_version',
    'num_sections',
    'numstrings',
    'printables',
    'sizeof_code',
    'sizeof_headers',
    'sizeof_heap_commit',
    'string_entropy',
    'symbols',
    'vsize',
    'size'
]

infeasible_features_pdf = [
    'createdate_ts',
    'createdate_tz',
    'moddate_ts',
    'moddate_tz',
    'version',
    # From this point on: there seems to be a problem with the Mimicus
    # tool. The modification is actually present in the file but when the file
    # is read back, the feature value appears unchanged
    # 'creator_dot',
    # 'creator_lc',
    # 'creator_num',
    # 'creator_oth',
    # 'creator_uc',
    # 'producer_dot',
    # 'producer_lc',
    # 'producer_num',
    # 'producer_oth',
    # 'producer_uc'
]

infeasible_features_drebin = [
    'api_permissions',
    'interesting_calls',#dangerous calls
    'api_calls',
    'urls'
]

features_to_exclude = {
    'ember': infeasible_features,
    'pdf': infeasible_features_pdf,
    'drebin': infeasible_features_drebin
}

# Criteria for the attack strategies
sample_selection_criteria = {'p-value','instance'}
