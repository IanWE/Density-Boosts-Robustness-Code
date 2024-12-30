import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def get_from_log(log):
    dict_pull_to_count = {x:0 for x in range(1, 61)}
    with open(log, 'r') as fp:
        for line in fp:
            if '### Evade!' in line:
                line = line.strip()
                pull = int(line.split(' ')[-1][:-1])
                for idx in range(pull, 61):
                    dict_pull_to_count[idx] += 1
    print(dict_pull_to_count)
    list_rate = []
    for idx in range(1, 61):
        list_rate.append(dict_pull_to_count[idx]/100)
    return list_rate

def main():
    atk = 'MAB'#sys.argv[1]
    namedict = {'remote0':'BaseNN','remote1':'LGBM','remote2':'LTNN','remote3':'BinarizedNN','remote4':'SCNN','remote5':'HistNN','remote6':'SCBNN',"remote7":"SCBDBNN","remote8":"SCBCropNN","remote9":"SCBMIXNN"}
    base_dir = '/root/MAB-malware/results_100/'
    rate_list = []
    #x = np.array(range(len(y1_list_rate)))
    x = np.array(range(60))#range(len(y1_list_rate)))
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    rect = fig.patch
    #rect.set_facecolor('none')
    rect.set_alpha(0)
    ax.set_facecolor("none")
    dirs = os.listdir(base_dir)
    for av in namedict:
        ds = list(filter(lambda x:av in x,dirs))
        if len(ds)==0:
            continue
        y_list_rate = np.array([0]*60).astype('float')
        for d in ds:
            y_list_rate += np.array(get_from_log(base_dir+f'{d}/rewriter.log'))
            #print(d,y_list_rate)
        print(ds,y_list_rate)
        y_list_rate /= len(ds)
        rate_list.append(y_list_rate)
        ax.plot(x, y_list_rate, label=namedict[av])
    plt.xlabel('Total number of attempts', fontsize=15)
    plt.ylabel('Evasion rate %', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.15),ncol=4, fontsize=10, frameon=False)
    #fig.subplots_adjust(wspace=0.1)
    
    #plt.show()
    #matplotlib.pyplot.title(av);
    plt.savefig("evasion_rate_100.pdf")

if __name__ == '__main__':
    main()

