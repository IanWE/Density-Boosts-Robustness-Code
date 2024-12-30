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
warnings.filterwarnings('ignore')
from core import attack_utils
import matplotlib.pyplot as plt
import ember
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
features, feature_names, name_feat, feat_name = data_utils.load_features(constants.features_to_exclude['ember'],'ember')

import lief
import sys
import io

saved_dir = os.path.join(constants.SAVE_MODEL_DIR,"ember")
#model = model_utils.load_model('lightgbm', 'ember',saved_dir,"base_lightgbm")

def extract_features(filename):
    #%%capture
    origin_output = sys.stdout
    origin_err = sys.stderr
    fake_stdout = io.StringIO()
    sample_path = os.path.join(constants.EMBER_MALWARE_DIR,i)
    print("processing",sample_path)
    sys.stdout = fake_stdout
    sys.stderr = fake_stdout
    with open(sample_path,"rb") as f:
        bytez = f.read()
    f = extractor.feature_vector(bytez)
    sys.stdout = origin_output
    sys.stderr = origin_err
    return f

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "-file",
        help="Path fot the sample",
        type=str,
        required=True
    )
    arguments = parser.parse_args()

    args = vars(arguments)
    

