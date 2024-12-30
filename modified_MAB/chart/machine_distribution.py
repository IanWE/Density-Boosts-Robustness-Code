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
        list_rate.append(dict_pull_to_count[idx]/1000)
    return list_rate

def main():
    av = 'MAB'#sys.argv[1]
    namedict = {'remote0':'Vanilla','remote1':'LGBM','remote2':'LT','remote3':'Binarization','remote4':'SC','remote5':'HistNN','remote6':'SCB',"remote7":"SCB-DB","remote8":"SCB-Crop","remote9":"SCB-DB-Crop"}
    base_dir = '/root/MAB-malware/results/'
    rate_list = []
    #x = np.array(range(len(y1_list_rate)))
    x = np.array(range(60))#range(len(y1_list_rate)))
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    rect = fig.patch
    #rect.set_facecolor('none')
    rect.set_alpha(0)
    ax.set_facecolor("none")
    for d in sorted(os.listdir("/root/MAB-malware/results/")):
        if av in d and '_8' not in d:
            y1_list_rate = get_from_log(f'/root/MAB-malware/results/{d}/rewriter.log')
            #print(len(x),len(y1_list_rate))
            print(d,len(y1_list_rate))
            classifier = d.split("_")[2]
            rate_list.append(y1_list_rate)
            ax.plot(x, y1_list_rate, label=namedict[classifier])
            #ax.plot(x, y1_list_rate, label=classifier)
    #ax.plot(x2, y2_list_rate, fillstyle='none', linewidth=1, label='TS')
    #ax.plot(x3, y3_list_rate, fillstyle='none', linewidth=1, color='red', label='TS')
    #axes = plt.gca()
    #axes.set_ylim([0, 1.05])
    #axes.set_xlim([0, 67])
    
    #plt.annotate(str(y1_list_rate[-1]), (60,y1_list_rate[-1]), color='red', fontsize=9)
    #plt.annotate(str(y2_list_rate[-1]), (60,y2_list_rate[-1]-3), color='red', textcoords="offset points", fontsize=9)
    #plt.annotate(str(y3_list_rate[-1]), (60,y3_list_rate[-1]-3), color='red', fontsize=9)

    plt.xlabel('Total number of attempts', fontsize=15)
    plt.ylabel('Evasion rate', fontsize=15)
    ax.legend(loc='lower right', ncol=2, fontsize=12)#, framealpha=0)
    fig.subplots_adjust(wspace=0.1)
    
    #plt.show()
    #matplotlib.pyplot.title(av);
    plt.savefig("evasion_rate_%s.pdf" %av)

if __name__ == '__main__':
    main()

