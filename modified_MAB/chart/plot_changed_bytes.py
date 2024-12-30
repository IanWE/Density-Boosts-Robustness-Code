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
        #'SCB-Crop': '/root/MAB-malware/results/output_MAB_remote8_16/minimal/',
        #'SCB-DB-Crop': '/root/MAB-malware/results/output_MAB_remote9_16/minimal/',
        }

def main():
    required_bytes = []
    avs = []
    for av, path in dict_av_to_path.items():
        print('='*40)
        print(av)
        avs.append(av)
        total_size = 0
        for i in os.listdir(path):
            total_size += os.path.getsize(path+"/"+i)-os.path.getsize("/root/MAB-malware/data/malware/"+i.split('.')[0])
        total_size /= len(os.listdir(path))
        #avg_added_bytes = calculate_average_size(path)
        #orig_bytes = calculate_average_size("/root/MAB-malware/data/malware/")
        #required_bytes.append(avg_added_bytes-orig_bytes)
        required_bytes.append(total_size)
        print('avg added bytes:', total_size)
    plot(avs,required_bytes)

def plot(avs,y):
    x = np.arange(len(y))
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    fig.subplots_adjust(wspace=0.5)
    rect = fig.patch
    rect.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_yscale('log')
    plt.grid(None)
    plt.bar(x, y)
    plt.xticks(x, avs, rotation=35)
    #ax.legend(loc='lower right', ncol=2, fontsize=12)#, framealpha=0)
    #plt.show()
    #matplotlib.pyplot.title(av);
    plt.tight_layout()
    plt.gcf().set_size_inches(8, 5)  
    #plt.xlabel('Total number of attempts', fontsize=12)
    plt.ylabel('Average changed bytes', fontsize=12)
    plt.savefig("changed_bytes.pdf" )
    
def calculate_average_size(folder_path):
    total_size = 0
    num_files = 0

    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件大小并累加到总大小中
            total_size += os.path.getsize(file_path)
            num_files += 1

    # 计算平均大小
    if num_files > 0:
        average_size = total_size / num_files
        return average_size
    else:
        return 0

if __name__ == '__main__':
    main()
