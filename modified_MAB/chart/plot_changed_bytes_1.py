import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils import *

dict_av_to_path = {
        'Vanilla': '/root/MAB-malware/results/output_MAB_remote0/minimal/',
        'LGBM': '/root/MAB-malware/results/output_MAB_remote1/minimal/',
        'LT': '/root/MAB-malware/results/output_MAB_remote2/minimal/',
        'Binarization': '/root/MAB-malware/results/output_MAB_remote3/minimal/',
        'SC': '/root/MAB-malware/results/output_MAB_remote4_16/minimal/',
        'SCB': '/root/MAB-malware/results/output_MAB_remote6_16/minimal/',
        'SCB-DB': '/root/MAB-malware/results/output_MAB_remote7_16/minimal/',
        'SCB-Crop': '/root/MAB-malware/results/output_MAB_remote8_16/minimal/',
        #'SCB-DB-Crop': '/root/MAB-malware/results/output_MAB_remote9_16/minimal/',
        }

def main():
    for av, path in dict_av_to_path.items():
        print('='*40)
        print(av)
        list_diff_count = []
        print(path)
        list_exe = os.listdir(path)
        print(len(list_exe))
        less_1_dict_action_to_exe = {}
        less_100_evade_count = 0
        less_1_evade_count = 0
        for exe in list_exe:
            sha256 = exe.split('.')[0]
            list_action = [x for x in exe.split('.') if len(x) == 2 or (len(x) == 3 and x != 'exe')]
            #print(list_action)
            diff_count = 0
            exe_path_ori = './data/malware/' + sha256
            exe_path = path + exe
            fp1 = open(exe_path_ori, 'rb')
            fp2 = open(exe_path, 'rb')
            content1 = fp1.read()
            content2 = fp2.read()
            for idx, byte in enumerate(content2):
                if idx >= len(content1) or byte != content1[idx]:
                    diff_count += 1
            fp1.close()
            fp2.close()
            #if diff_count < 100:
            if diff_count < 10:
                print(exe, diff_count)
            if diff_count < 100:
                less_100_evade_count += 1
            if diff_count == 1:
                less_1_evade_count += 1
                action = list_action[0]
                if action not in less_1_dict_action_to_exe:
                    less_1_dict_action_to_exe[action] = []
                less_1_dict_action_to_exe[action].append(exe)
            list_diff_count.append(diff_count)
        list_diff_count.sort()
        print(list_diff_count)
        #with open('%s' %av, 'w') as fp:
        #    for diff in list_diff_count:
        #        fp.write('%d\n' %diff)
        print('less_100_evade_count:', less_100_evade_count)
        print('less_1_evade_count:', less_1_evade_count)
        for action, list_exe in less_1_dict_action_to_exe.items():
            print(action, len(list_exe), list_exe)
        plot(list_diff_count, av)

def plot(y,av):
    x = np.arange(len(y))
    fig, ax = plt.subplots()
    ax.set_yscale('log')

    plt.xlabel('evasive samples %d - %d' %(1, len(y)), fontsize=12)
    plt.ylabel('change byte amount', fontsize=12)

    plt.bar(x, y, color='silver')
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    fig.subplots_adjust(bottom=0.5)
    #plt.show()
    plt.savefig('./images/byte_%s.pdf' %av.lower())

if __name__ == '__main__':
    main()
