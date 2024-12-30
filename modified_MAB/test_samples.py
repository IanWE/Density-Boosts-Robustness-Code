from utils import *
import random
import pefile
import mmap
import copy
import hashlib
from scipy.stats import beta
import pexpect
import string
#from manipulate2 import *
import json
import requests
from models import *
import time

#MALCONV_MODEL_PATH = 'models/malconv/malconv.checkpoint'
#model = MalConvModel( MALCONV_MODEL_PATH, thresh=0.5 )
#EMBER_2019_MODEL_PATH = 'models/ember_2019/ember_model.txt'
#model = EmberModel_2019( EMBER_2019_MODEL_PATH, thresh=0.8336 )

#for folder in ['results/output_MAB_remote0/minimal/','results/output_MAB_remote1/minimal/','results/output_MAB_remote2/minimal/','results/output_MAB_remote3/minimal/','results/output_MAB_remote4/minimal/','results/output_MAB_remote5_16/minimal/','results/output_MAB_remote7_bundle/minimal/','results/output_MAB_remote9_bundle/minimal/']:
for folder in ['results/output_MAB_remote13_bundledbpad/evasive/']:
    for i in range(1):
        print(f'Folder {folder} {i}')
        model = RemoteModel('remote',i)
        list_f = os.listdir(folder)
        total = len(list_f)
        evaded = detect = 0
        for f in list_f:
            eva = model.is_evasive('%s%s' %(folder, f))
            if eva:
                evaded += 1
            else:
                detect += 1
            print('%d/%d/%d' %(evaded, detect, total))
        print('######################')
        with open('transferability.txt','a') as ff:
            ff.write(f'Folder {folder} to {i}: {evaded}/{total}\n')

