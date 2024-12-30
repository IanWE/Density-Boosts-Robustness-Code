import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils import *

dict_av_to_path = {
        'VanillaNN': '/root/MAB-malware/results/output_MAB_remote0/minimal/',
        'LightGBM': '/root/MAB-malware/results/output_MAB_remote1/minimal/',
        'LTNN': '/root/MAB-malware/results/output_MAB_remote2/minimal/',
        'BinarizedNN': '/root/MAB-malware/results/output_MAB_remote3/minimal/',        
        'HistogramNN': '/root/MAB-malware/results/output_MAB_remote4/minimal/',
        'SCNN': '/root/MAB-malware/results/output_MAB_remote5_16/minimal/',
        #'SCBNN': '/root/MAB-malware/results/output_MAB_remote6_True/minimal/',
        'SCBNN': '/root/MAB-malware/results/output_MAB_remote7_bundle/minimal/',
        #'SCBNN-DBM': '/root/MAB-malware/results/output_MAB_remote8_dbm/minimal/',
        #'SCBNN-DB': '/root/MAB-malware/results/output_MAB_remote8_True/minimal/',
        'SCBNN-DB': '/root/MAB-malware/results/output_MAB_remote9_bundle/minimal/',
        #'SCBundleNN-DB': '/root/MAB-malware/results/output_MAB_remote9_bundle/minimal/',
        'SCBNN-DB-PAD': '/root/MAB-malware/results/output_MAB_remote12_bundledbfilter/minimal/',
        'SCBNN-PAD': '/root/MAB-malware/results/output_MAB_remote10/minimal/',
        'VanillaNN-PAD': '/root/MAB-malware/results/output_MAB_remote11filter/minimal/',
        }

def action_rename(action_name):
    action_name = action_name.replace('O1', 'OA1')
    action_name = action_name.replace('S1', 'SA1')
    action_name = action_name.replace('SR1', 'SR1')
    action_name = action_name.replace('P1', 'SP1')
    action_name = action_name.replace('CP', 'SP1')
    return action_name

dict_av_to_min = {
        'VanillaNN': 1,
        'LightGBM': 0,
        'LTNN': 1,
        'BinarizedNN': 1,
        'HistogramNN': 0,
        'SCNN': 0,
        'SCBNN': 1,
        'SCBundleNN': 1,
        'SCBNN-DB': 1,
        'SCBNN-DB-PAD': 1,
        'SCBNN-DBM': 1,
        'SCBundleNN-DB': 1,
        'SCBNN-RD': 1,
        'SCBNN-COMBINED': 2,
        'SCBNN-PAD': 1,
        'VanillaNN-PAD': 1,
        }

plt.style.use('bmh')
def main():
    for av, path in dict_av_to_path.items():
        dict_action_list_to_count = {}
        print('='*40)
        print(av)
        list_action = []
        print(path)
        list_exe = os.listdir(path)
        print(len(list_exe))
        for exe in list_exe:
            list_action = [action_rename(x) for x in exe.split('.') if len(x) == 2 or (len(x) == 3 and x != 'exe')]
            list_action.sort()
            #if len(list_action) > 3:
            #    continue
            list_action = str(list_action).replace('\'', '').replace(' ', '')
            if list_action not in dict_action_list_to_count:
                dict_action_list_to_count[list_action] = 0
            dict_action_list_to_count[list_action] += 1

        #listofTuples = sorted(dict_action_list_to_count.items(), key=lambda x:(len(x[0]), -x[1]))
        dict_larger = {}
        total_count = 0
        for k,v in dict_action_list_to_count.items():
            total_count += v
        for k,v in dict_action_list_to_count.items():
            if v > dict_av_to_min[av]:       # debug!
                dict_larger[k] = v/total_count * 100
        listofTuples = sorted(dict_larger.items(), key=lambda x:(-x[1]))[:10]
        #listofTuples = sorted(listofTuples, key=lambda x:len(x[0]))

        y = []
        label = []
        for elem in listofTuples :
            #print(elem[0], elem[1] )
            y.append(elem[1])
            label.append(elem[0])
        label_processed = []
        for l in label:
            p = []
            ll = l[1:-1].split(',')
            c = 1
            pre = '' 
            for ac in ll:
                if ac == pre or pre == '':
                    c = c + (ac == pre)
                    pre = ac
                    continue
                elif c>1:
                    p.append(pre+'*'+str(c))
                else:
                    p.append(pre)
                pre = ac
                c = 1
            if c > 1:
                p.append(pre+'*'+str(c))
            else:
                p.append(pre)
            label_processed.append('['+','.join(p)+']')
        x = np.arange(len(y))
        fig = plt.figure(figsize=(6,3)) #创建绘图对象
        ax = fig.add_subplot(111)
        plt.style.use('bmh')
        plt.grid(False)
        plt.grid(axis='y')
        #plt.grid(False)
        ax.patch.set_alpha(0)
        ax.bar(x, y, hatch='////')

        #ax.set_xlabel("Action combination",fontsize=15)
        #ax.xaxis.set_tick_params(color='black', labelcolor='black')
        if av == 'VanillaNN' or av == 'HistogramNN' or av=='SCBNN-DB-PAD':
            ax.set_ylabel('Percentage %',fontsize=15)
        plt.xticks(x, label_processed,rotation=45, fontsize=12,ha='right',rotation_mode='anchor')
        plt.yticks(np.arange(0,101,20),fontsize=15)
        fig.subplots_adjust(bottom=0.3)
        #fig.subplots_adjust(top=0.33)
        plt.draw()
        #plt.savefig('./images/combination_%s_slanted.eps' %av.lower())
        plt.savefig('./images/combination_%s_slanted.pdf' %av.lower())#,bbox_inches='tight')

if __name__ == '__main__':
    main()
