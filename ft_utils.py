import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import time
import datetime
import glob
import json
import subprocess
import math
import requests
import slackweb

JSONFILE = os.path.join(os.path.expanduser('~'),'token.json')
LINE_NOTIFY_URL = 'https://notify-api.line.me/api/notify'

if os.name == 'posix':
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    NEWLINECODE = '\n'
elif os.name == 'nt':
    import matplotlib.pyplot as plt
    NEWLINECODE = '\r\n'
else:
    raise OSError('Unknown OS.')

def act_unix(command):
    if not isinstance(command, str):
        raise TypeError('Unix command is str type.')

    return_code = subprocess.call(command.split())
    assert return_code == 0,\
            'Not activate [{}]'.format(command)

def mkdirs(dirpath, response=True):
    if not os.path.exists(dirpath):
        if os.name == 'posix':
            command = 'mkdir -p {}'.format(dirpath)
        elif os.name == 'nt':
            command = 'mkdir {}'.format(dirpath)
        else:
            raise OSError('Not create [{}]'.format(dirpath))

        act_unix(command)

    if response:
        return dirpath

def datasets_check(dirpath=None, savefile=None):
    data_dirs = os.listdir(dirpath)

    data_dict = {'name':[],'class':[]}
    for c in data_dirs:
        data_list = glob.glob(os.path.join(dirpath,c,'*.jpg'))
        for img in data_list:
            imgname = os.path.basename(img).split('.jpg')[0]

            data_dict['name'].append(imgname)
            data_dict['class'].append(c)

    class_df = pd.DataFrame(data_dict)
    class_df.sort_values(by='name', inplace=True)

    if savefile is not None:
        filename = os.path.basename(savefile)
        if '.csv' in filename:
            class_df.to_csv(savefile)
        elif '.xlsx' in filename:
            try:
                class_df.to_excel(savefile)
            except Exception as err:
                sys.stdout.write(str(err)+NEWLINECODE)
                sys.stdout.write('Please install openpyxl via pip.'+NEWLINECODE)
        elif '.html' in filename:
            class_df.to_html(savefile)
        else:
            raise Exception('Incorrect file extension.')

class KerasVisualizer(object):
    def __init__(self, savedir, font_size=8, graph_tick='in',
            grid=True, legend_curve_edge=False, dpi=100):
        plt.rcParams['font.size'] = font_size
        #plt.rcParams['font.family'] = 'sans-serif'
        #plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['xtick.direction'] = graph_tick
        plt.rcParams['ytick.direction'] = graph_tick
        plt.rcParams['xtick.major.width'] = 1.2
        plt.rcParams['ytick.major.width'] = 1.2
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.grid'] = grid
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.linewidth'] = 0.3
        plt.rcParams['legend.markerscale'] = 2
        plt.rcParams['legend.fancybox'] = legend_curve_edge
        plt.rcParams['legend.framealpha'] = 1
        plt.rcParams['legend.edgecolor'] = 'black'

        self.savedir = mkdirs(savedir)
        self.dpi = dpi

    def keras_histsave(self, history, indicator, showinfo=True):
        train_indices = history.history[indicator]
        val_indices = history.history['val_{}'.format(indicator)]

        img_name = '{}.jpg'.format(indicator)

        plt.plot(train_indices, label='train-{}'.format(indicator), color='r', lw=2)
        plt.plot(val_indices, label='val-{}'.format(indicator), color='g', lw=2)

        if showinfo:
            plt.xlabel('Iterations')
            plt.ylabel('{}'.format(indicator.capitalize())) # capitalizeは最初だけ大文字にできる
            plt.title('Learning Curve for {}'.format(indicator.capitalize()))

        plt.legend()
        plt.savefig(os.path.join(self.savedir, img_name), dpi=self.dpi)
        plt.close() # これ付けないと学習指標が全て一つの画像になってしまう

    def keras_probability_hist(self, probs, showinfo=True):
        plt.hist(pd.Series(probs), lw=5, bins=100, color='g', normed=True, ec='black')

        if showinfo:
            plt.xlabel('Max Probability')
            plt.ylabel('Normalized Frequency')
            plt.title('Frequency for Max Probability')

        plt.savefig(os.path.join(self.savedir, 'hist.jpg'), dpi=self.dpi)

    def middle_layer_visualizer(self, model_matrix=None, img_name=None):
        assert (model_matrix is not None), 'Not found model matrix.'
        assert (img_name is not None), 'Input save heatmap images.'

        for i, (layer_name,activation) in enumerate(model_matrix):
            try:
                num_channel = activation.shape[3] # 四次元テンソルじゃないとエラーになる
            except IndexError:
                raise IndexError('Not 4D tensor. Shape is {}.'.format(activation.shape))

            cols = math.ceil(math.sqrt(num_channel)) # ceilで小数点以下の切り上げを行う
            rows = math.floor(num_channel/cols) # floorで小数点以下の値の切り下げを行う

            screen = list()
            for y in range(rows):

                row = list()
                for x in range(cols):
                    j = y*cols+x

                    if j < num_channel:
                        row.append(activation[0,:,:,j])
                    else:
                        row.append(np.zeros())

                screen.append(np.concatenate(row,axis=1))

            screen = np.concatenate(screen,axis=0)

            plt.figure()
            sns.heatmap(screen, xticklabels=False, yticklabels=False)
            plt.savefig(os.path.join(self.savedir, '{}_heatmap_{}.jpg'.format(
                layer_name,img_name)),dpi=self.dpi)

            plt.close()

class Timer:
    def __init__(self, api='line', output=True):
        if not os.path.exists(JSONFILE):
            raise FileNotFoundError('No JSON file.')

        with open(JSONFILE, 'r') as t:
            tokens = json.load(t)

            if api == 'slack':
                self.token = tokens['slack_token']
                self.s = slackweb.Slack(url=self.token)
            elif api == 'line':
                self.token = tokens['line_token']
                self.s = None
            else:
                raise ValueError('API : slack or LINE')

        self.api = api
        self.output = output

    def timer(self, func):
        def __wrapper(*args, **kwargs):
            start = time.time()
            activated_now = datetime.datetime.now()

            res = func(*args, **kwargs)

            s = time.time()-start
            m = s/60
            h = m/60
            d = h//24
            msg = '''

[{}]
Function   : [{}]
Activation :
    {:.3f}\t[sec]
    {:.3f}\t[min]
    {:.3f}\t[hour]
    {:.3f}\t[day]

            '''.format(
                    activated_now,
                    func.__name__,
                    s,m,h,d)

            if self.api == 'slack':
                self.__send_to_slack(msg)
            elif self.api == 'line':
                self.__send_to_line(msg)

            if self.output:
                sys.stdout.write(msg+NEWLINECODE)

            if res is not None:
                return res

        return __wrapper

    def __send_to_slack(self, msg):
        try:
            self.s.notify(text=msg)
        except Exception as err:
            sys.stdout.write(str(err)+NEWLINECODE)

    def __send_to_line(self, msg):
        payload = {'message':msg}
        headers = {'Authorization':'Bearer '+self.token}

        try:
            response = requests.post(
                    LINE_NOTIFY_URL, data=payload, headers=headers
                    )
        except Exception as err:
            sys.stdout.write(str(err)+NEWLINECODE)
