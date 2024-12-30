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

plt.style.use('bmh')
def main():
    av = 'MAB'#sys.argv[1]
    namedict = {'remote0':'VanillaNN','remote1':'LightGBM','remote2':'LTNN','remote3':'BinarizedNN','remote4':'HistogramNN','remote5':'SCNN','remote7':'SCBNN',"remote9":"SCBNN-DB"}#,"remote10":"SCBNN-PAD"}
    #namedict = {"remote10filter":"SCBNN-PAD","remote11filter":"VanillaNN-PAD"}#,"remote10":"SCBNN-PAD"}
    base_dir = '/root/MAB-malware/results/'
    rate_list = []
    #x = np.array(range(len(y1_list_rate)))
    x = np.array(range(60))#range(len(y1_list_rate)))
    plt.subplots_adjust(wspace=0.1)
    fig = plt.figure(figsize=(6,4)) #创建绘图对象
    plt.subplots_adjust(wspace=0.1)
    ax = fig.add_subplot(111)
    plt.style.use('bmh')
    plt.grid(False)
    plt.grid(axis='y')
    ax.patch.set_alpha(0)
    for d in sorted(os.listdir("/root/MAB-malware/results/")):
        if av in d:
            y1_list_rate = get_from_log(f'/root/MAB-malware/results/{d}/rewriter.log')
            #print(len(x),len(y1_list_rate))
            print(d,len(y1_list_rate))
            classifier = d.split("_")[2]
            rate_list.append(y1_list_rate)
            if classifier in namedict:
                ax.plot(x, y1_list_rate, label=namedict[classifier])
            #ax.plot(x, y1_list_rate, label=classifier)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.xlabel('Total number of attempts', fontsize=15)
    #plt.ylabel('Attack success rate', fontsize=15)
    ax.set_xlabel("Total number of attempts",fontsize=15)
    ax.set_ylabel('Attack success rate',fontsize=15)
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1.05),fontsize=11,facecolor="none")
    plt.tight_layout()
    ax.legend(fontsize=11,facecolor="none",ncol=2)
    plt.savefig("evasion_rate_%s.pdf" %av)
    #plt.savefig("evasion_rate_pad.pdf")

if __name__ == '__main__':
    main()

