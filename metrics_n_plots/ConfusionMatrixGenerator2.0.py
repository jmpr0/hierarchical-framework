#!/usr/bin/python3

import numpy as np, copy
import sys, getopt, os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from scipy.stats.mstats import gmean
from imblearn.metrics import geometric_mean_score
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import seaborn as sns
import itertools

pd.options.mode.chained_assignment = None  # default='warn'


def draw_square(ax, top_left_coordinates, side, ls='solid', lw=1, ec='black'):
    '''
    Draw a square given as input top-left coordinates [tuple] and side length [float] with a certain ls (linestyle) [string].
    '''

    ax.add_patch(
        patches.Rectangle(top_left_coordinates, side, side, fill=False, linestyle=ls, linewidth=lw, edgecolor=ec))


def main(argv):
    np.random.seed(0)

    input_file = ''

    classifier_name = 'wrf'

    try:
        opts, args = getopt.getopt(argv, "hi:c:", "[input_file=]")
    except getopt.GetoptError:
        print('ConfusionMatrixGenerator.py -i <input_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('ConfusionMatrixGenerator.py -i <input_file>')
            sys.exit()
        if opt in ("-i", "--input_file"):
            input_file = arg
        if opt in ("-c", "--clf"):
            classifier_name = arg

    print('\n' + input_file + ' elaboration...\n')
    # load .dat file
    file = open(input_file, 'r')
    lines = file.readlines()
    file.close()

    file_name = os.path.basename(input_file)

    level = int([file_name.split('_')[i + 1] for i in range(0, len(file_name.split('_'))) if
                 file_name.split('_')[i] == 'level'][0])

    oracles = list()
    predictions = list()

    classified_ratios = list()

    save_cr = False

    fold_cnt = -1  # first iteration make fold_cnt -eq to 0
    cr_cnt = -1
    for line in lines:
        if '@fold_cr' not in line:
            if not save_cr and '@fold' not in line:
                oracles[fold_cnt].append(line.split()[0])
                predictions[fold_cnt].append(line.split()[1])
            elif '@fold' in line:
                oracles.append(list())
                predictions.append(list())
                fold_cnt += 1
            else:
                classified_ratios[cr_cnt] = float(line)
                save_cr = False
                oracles.append(list())
                predictions.append(list())
                fold_cnt += 1
        else:
            classified_ratios.append(0)
            save_cr = True
            cr_cnt += 1

    k = fold_cnt + 1

    confusion_matrices = list()
    tags = list(set([o for oracle in oracles for o in oracle]))

    for i in range(0, k):
        confusion_matrices.append(confusion_matrix(oracles[i], predictions[i], tags))
        confusion_matrices[i] = confusion_matrices[i].astype('float') / confusion_matrices[i].sum(axis=1)[:, np.newaxis]

    # Row with at most one NaN value are set as 0-values vector
    # confusion_matrices[i] = np.asarray([ [0]*len(x) if np.isnan(x).any() else x for x in confusion_matrices[i] ])

    cm = np.zeros(shape=confusion_matrices[0].shape)

    for r in range(0, cm.shape[0]):
        for c in range(0, cm.shape[1]):
            temp_vector = list()
            for j in range(0, k):
                temp_vector.append(confusion_matrices[j][r, c])

            temp_vector_noNaN = [t for t in temp_vector if not np.isnan(t)]
            cm[r, c] = np.mean(temp_vector_noNaN)

    image_folder = './image_' + classifier_name + '/confusion_matrix/'

    if not os.path.exists('./image_' + classifier_name):
        os.makedirs('./image_' + classifier_name)
        os.makedirs(image_folder)
    elif not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # if level==1:
    # 	dict_tags_abc = {
    # 		'1.0':'MALICIOUS',
    # 		'0.0':'BENIGN'
    # 	}
    # elif level==2:
    # 	dict_tags_abc = {
    # 		'DDoS':'DDoS',
    # 		'DoS':'DoS',
    # 		'Normal':'BENIGN',
    # 		'Reconnaissance':'Scan',
    # 		'Theft':'Theft'
    # 	}
    # elif level==3:
    # 	dict_tags_abc = {
    # 		'Data_Exfiltration':'Exfilration',
    # 		'DDoS_HTTP':'DDoS HHTP',
    # 		'DoS_HTTP':'DoS HTTP',
    # 		'Keylogging':'Keylogging',
    # 		'Normal':'BENIGN',
    # 		'OS_Fingerprint':'OS Scan',
    # 		'Service_Scan':'Service Scan'
    # 	}

    # if level==1:
    # 	dict_tags_abc = {
    # 		'Tor':'a',
    # 		'JonDonym':'c',
    # 		'I2P':'b'
    # 	}
    # elif level==2:
    # 	dict_tags_abc = {
    # 		'TorPT':'c',
    # 		'TorApp':'b',
    # 		'I2PApp80BW':'e',
    # 		'Tor':'a',
    # 		'I2PApp0BW':'d',
    # 		'JonDonym':'g',
    # 		'I2PApp':'f'
    # 	}
    # elif level==3:
    # 	dict_tags_abc = {
    # 		'I2PSNARK_App80BW':'m',
    # 		'IRC_App0BW':'k',
    # 		'Eepsites_App0BW':'l',
    # 		'I2PSNARK_App0BW':'j',
    # 		'ParticipatingTunnels_App':'t',
    # 		'Eepsites_App':'r',
    # 		'Tor':'a',
    # 		'IRC_App80BW':'n',
    # 		'I2PSNARK_App':'p',
    # 		'FTE':'f',
    # 		'Obfs3':'h',
    # 		'Eepsites_App80BW':'o',
    # 		'ExploratoryTunnels_App':'s',
    # 		'Meek':'g',
    # 		'JonDonym':'u',
    # 		'IRC_App':'q',
    # 		'Flashproxy':'e',
    # 		'Torrent':'c',
    # 		'scramblesuit':'i',
    # 		'Browsing':'d',
    # 		'Streaming':'b'
    # 	}

    if level == 1:
        dict_tags_abc = {
            'Video': 'a',
            'Generic': 'c',
            'Background': 'b'
        }
    elif level == 2:
        dict_tags_abc = {
            'OnDemand': 'c',
            'CloudVR': 'b',
            'Music\&Audio': 'e',
            'Chat': 'a',
            'Short': 'd',
            'Social': 'g',
            'Shopping': 'f',
            'Travel\&Local': 'h',
            'Background': 'i'
        }
    elif level == 3:
        dict_tags_abc = {
            'Whatsapp': 'a',
            'Skype': 'b',
            'Messenger': 'c',
            'Within': 'd',
            'DiscoveryVR': 'e',
            'FulldiveVR': 'g',
            'PrimeVideo': 'h',
            'Netflix': 'i',
            'Vimeo': 'j',
            'Instagram': 'k',
            'TikTok': 'l',
            'Snapchat': 'm',
            'Musixmatch': 'n',
            'Spotify': 'o',
            'SoundCloud': 'p',
            'Wish': 'q',
            'Amazon': 'r',
            'eBay': 's',
            'Linkedin': 't',
            'Pinterest': 'u',
            'Tumblr': 'v',
            'Booking': 'w',
            'TripAdvisor': 'x',
            'FourSquare': 'y',
            'Background': 'z',
        }

    tags_abc = np.vectorize(dict_tags_abc.get)(tags)

    indexes = np.argsort(tags_abc)
    tags = [tags[i] for i in indexes]
    tags_abc = [tags_abc[i] for i in indexes]

    cm = np.array([cm[i, :] for i in indexes])
    for i in range(0, cm.shape[0]):
        cm[i] = [cm[i, j] for j in indexes]

    # Target confusion-matrix
    df_cm = pd.DataFrame(cm, index=[tag for tag in tags_abc], columns=[tag for tag in tags_abc])
    plt.clf()
    fig = plt.figure(1)

    ax = sns.heatmap(df_cm * 100, vmin=0.001, vmax=100, cmap='Blues', square=True,
                     norm=LogNorm(cm.min() * 100, cm.max() * 100), cbar=False)

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_ticks([0.001, 0.01, 0.1, 1, 10, 100])
    cbar.set_ticklabels([0.001, 0.01, 0.1, 1, 10, 100])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(bottom=False, top=False, left=False, right=False, labelsize=13)

    plt.xlabel('Predicted Class', fontsize=18)
    plt.ylabel('Actual Class', fontsize=18)
    # plt.tick_params(bottom=False,top=False,left=False,right=False,labelrotation=0,labelsize=13)
    plt.tick_params(bottom=False, top=False, left=False, right=False, labelsize=13)

    plt.tight_layout(rect=[0, 0.01, 1, 1])

    # Squares for Anon17
    # L2 squares
    # Normal Tor square
    draw_square(ax, (0, 0), 1, lw=2)
    # Tor Apps
    draw_square(ax, (1, 1), 3, lw=2)
    # Tor PT
    draw_square(ax, (4, 4), 5, lw=2)
    # I2P 80BW
    draw_square(ax, (9, 9), 3, lw=2)
    # I2P 0BW
    draw_square(ax, (12, 12), 3, lw=2)
    # I2P Apps
    draw_square(ax, (15, 15), 5, lw=2)
    # JD
    draw_square(ax, (20, 20), 1, lw=2)

    # L1 squares
    # TOR square
    draw_square(ax, (0, 0), 9, ls='dashed', lw=2, ec='black')
    # I2P square
    draw_square(ax, (9, 9), 11, ls='dashed', lw=2, ec='black')
    # JD square
    draw_square(ax, (20, 20), 1, ls='dashed', lw=2, ec='black')

    plt.savefig(image_folder + file_name[:-4] + '_confusion_matrix.eps', format='eps', dpi=300, bbox_inches='tight')

    print(len([cm[r, c] for r in range(0, cm.shape[0]) for c in range(0, cm.shape[1])]),
          len([cm[r, c] for r in range(0, cm.shape[0]) for c in range(0, cm.shape[1]) if cm[r, c] >= 0.001]) / len(
              [cm[r, c] for r in range(0, cm.shape[0]) for c in range(0, cm.shape[1])]))


if __name__ == "__main__":
    main(sys.argv[1:])
