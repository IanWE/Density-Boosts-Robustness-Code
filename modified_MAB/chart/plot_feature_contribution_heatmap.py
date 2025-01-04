import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



dict_av_to_path = {
        'VanillaNN': '/root/MAB-malware/results/output_MAB_remote0/minimal/',
        'LightGBM': '/root/MAB-malware/results/output_MAB_remote1/minimal/',
        'LTNN': '/root/MAB-malware/results/output_MAB_remote2/minimal/',
        'BinarizedNN': '/root/MAB-malware/results/output_MAB_remote3/minimal/',        
        'HistogramNN': '/root/MAB-malware/results/output_MAB_remote4/minimal/',
        'SCNN': '/root/MAB-malware/results/output_MAB_remote5_16/minimal/',
        'SCBNN': '/root/MAB-malware/results/output_MAB_remote6/minimal/',
        'SCBNN-DB': '/root/MAB-malware/results/output_MAB_remote7_16/minimal/',
        'SCBNN-RD': '/root/MAB-malware/results/output_MAB_remote8_16/minimal/',
        'SCBNN-COMBINED': '/root/MAB-malware/results/output_MAB_remote9/minimal/',
        }

dict_action_list_to_count = {}
dict_action_to_feature = {
        'OA1': 'file hash',
        'SP1': 'section hash',
        'SA1': 'section count',
        'SR1': 'section name',
        'CP1': 'section hash',
        'OA': 'byte entropy',
        'SP': 'section padding',
        'SA': 'byte entropy',
        'SR': 'section name',
        'RC': 'certificate',
        'RD': 'debug',
        'BC': 'checksum',
        'CR': 'code seq' 
        }

def get_display_name(av):
    if 'kaspersky' in av:
        display_name = 'AV4'
    elif 'bitdefender' in av:
        display_name = 'AV3'
    elif 'avast' in av:
        display_name = 'AV1'
    elif 'avira' in av:
        display_name = 'AV2'
    elif 'ember_2019' in av:
        display_name = 'EMBER'
    elif 'ember_2019' in av:
        display_name = 'EMBER(T&A)'
    elif 'clamav' in av:
        display_name = 'ClamAV'
    return display_name

def main():
    list_values = []
    for av, path in dict_av_to_path.items():
        dict_feature_to_sha256 = {
                'file hash': set(),
                'section hash': set(),
                'section count': set(),
                'section name': set(),
                'section padding': set(),
                'debug': set(),
                'checksum': set(),
                'certificate': set(),
                'code seq': set(),
                'byte entropy': set(),
        }
        print('='*40)
        print(av)
        #list_action = []
        #print(path)
        list_exe = os.listdir(path)
        #print(len(list_exe))

        for exe in list_exe:
            sha256 = exe.split('.')[0]
            #print(sha256)
            #list_action = [x.replace('CP', 'C1').replace('RS', 'RC') for x in exe.split('.') if len(x) == 2]
            list_action = [x for x in exe.split('.') if len(x) == 2 or (len(x) == 3 and x != 'exe')]
            list_action.sort()
            for action in list_action:
                if action == 'OA1':
                    if len(list_action) == 1:
                        dict_feature_to_sha256[dict_action_to_feature[action]].add(sha256)
                else:
                    dict_feature_to_sha256[dict_action_to_feature[action]].add(sha256)

        dict_larger = {}
        total_amount = 0
        for k in dict_feature_to_sha256.keys():
            total_amount += len(dict_feature_to_sha256[k])
        for k in dict_feature_to_sha256.keys():
            dict_larger[k] = len(dict_feature_to_sha256[k])/total_amount * 100
        listofTuples = dict_larger.items()
        print(dict_larger)
        list_values.append(list(dict_larger.values()))

        #y = []
        #label = []
        #for elem in listofTuples :
        #    print(elem[0], elem[1] )
        #    y.append(elem[1])
        #    label.append(elem[0])
        #x = np.arange(len(y))
        #fig, ax = plt.subplots()
        #plt.ylabel('percentage %', fontsize=14)

        #plt.bar(x, y, color='silver')
        #plt.xticks(x, label, rotation=90, fontsize=14)
        #fig.subplots_adjust(bottom=0.5)
        ##plt.show()
        #plt.savefig('/home/wei/feature_%s.pdf' %get_display_name(av).lower())
    value_array = None
    features = list(dict_larger.keys())
    print(features)
    print(list_values)
    plot_feature(features, list_values)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel='', **kwargs):
    '''
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    '''

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, fraction=0.05, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom')

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha='right',
             rotation_mode='anchor')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0)
    ax.tick_params(which='minor', bottom=False, left=False)

    return im

def annotate_heatmap(im, data=None, valfmt='{x:.2f}',
                     textcolors=['black', 'white'],
                     threshold=None, **textkw):
    '''
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. '$ {x:.2f}', or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    '''

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment='center',
              verticalalignment='center')
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each 'pixel'.
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            #if i != j:
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
            #else:
            #    kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            #    text = im.axes.text(j, i, '-', **kw)
            #    texts.append(text)

    return texts

def plot_feature(features, list_values):
    avs = list(dict_av_to_path)#['EMBER', 'MalColnv', 'ClamAV', 'AV1', 'AV2', 'AV3']
    
    value_array = np.array(list_values)
    
    print(features)
    print(value_array)
    fig, ax = plt.subplots()
    
    im = heatmap(value_array, avs, features, ax=ax,
                       #cmap='YlOrBr')
                       cmap='Reds')
    texts = annotate_heatmap(im, valfmt='{x:.2f}%', fontsize=8)
    
    fig.tight_layout()
    fig.set_size_inches(7, 5.5)
    fig.subplots_adjust(top=0.85)
    #plt.show()
    plt.savefig('./feature_contribution_heatmap.pdf')

if __name__ == '__main__':
    main()
