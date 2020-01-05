from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from PIL import Image
from ft_utils import Timer, mkdirs
import numpy as np
import os
import sys
import glob

TIMER = Timer()

if os.name == 'posix':
    NEWLINECODE = '\n'
elif os.name == 'nt':
    NEWLINECODE = '\r\n'
else:
    raise OSError('Unknown OS')

class SamplePreprocessing(object):
    def __init__(self, datatype='mnist'):
        if datatype == 'mnist':
            (xlearn, ylearn),(xpred, ypred) = mnist.load_data()
            xlearn = xlearn[:,:,:,np.newaxis].astype('float32')/255.
            xpred = xpred[:,:,:,np.newaxis].astype('float32')/255.
        elif datatype == 'cifar10':
            (xlearn, ylearn),(xpred, ypred) = cifar10.load_data()
            xlearn = xlearn.astype('float32')/255.
            xpred = xpred.astype('float32')/255.
        else:
            raise ValueError('datatype : mnist/cifar10')

        ylearn = to_categorical(ylearn, len(set(ylearn)))
        ypred = to_categorical(ypred, len(set(ypred)))

class ClassifyPreprocessing(object):
    def __init__(self, learndir=None, preddir=None, height=56, width=56, mode='RGB'):
        '''
        目的：画像分類用の前処理クラスインスタンス

        引数：

            learndir : 学習用のデータセットディレクトリパス, デフォルトはNone
            preddir : 未知画像を予測用データとする場合でのディレクトリパス, デフォルトはNone
            height : リサイズ後の画像Y軸ピクセル, デフォルトは56
            width : リサイズ後の画像X軸ピクセル, デフォルトは56
            mode : 画像を行列変換する際の読み込みモード

                RGB : カラー画像, channelは3になる
                L : グレースケール画像, channelは1になる

        使い方：
            selfの付いた変数は属性として取得可能

                ex. 学習用データセットを取得したい
                    >>> dataset = ClassifyPreprocessing(...)
                    >>> img_dataset = dataset.X

        '''
        if learndir is None:
            raise TypeError('Learning Dataset Not Existed.')

        if mode == 'L':
            channel = 1
        elif mode == 'RGB':
            channel = 3
        else:
            raise ValueError('Not recoginize channel.')

        self.mode = mode
        self.datashape = (width,height,channel)

        self.X,self.Y,self.N = self.__data_loader(datasetdir=learndir)

        if preddir is not None:
            self.pred_x, self.pred_y, self.pred_n = self.__data_loader(
                    datasetdir=preddir)
        else:
            self.pred_x, self.pred_y, self.pred_n = None, None, None

    @TIMER.timer
    def unused_split(self, splittype='holdout', splits=0.2):
        '''
        目的：予測用データに未使用画像を使用する場合のデータセット分割メソッド

        引数：

            splittype : データセットの分割法

                holdout : ホールドアウト法による分割, この場合splitsはテスト画像の割合
                kfold : K-交差検証法による分割, この場合splitsはsplitする分割数kに相当

            splits : holdoutなら0から1のfloat, kfoldならint

        返り値：
            xtrain, ytrain : 訓練用の画像データと正解クラスベクトル
            xval, yval : 検証用の画像データと正解クラスベクトル
            xpred, ypred : 予測用の画像データと正解クラスベクトル
            npred : 予測用の画像データの名称を格納した配列

        '''
        if splittype == 'holdout':
            xtrain, xval, xpred, ytrain, yval, ypred, npred = self.__unused_holdout(
                    self.X, self.Y, self.N, splits=splits)
        elif splittype == 'kfold':
            xtrain, xval, xpred, ytrain, yval, ypred, npred = self.__unused_kfold_cv(
                    self.X, self.Y, self.N, splits=splits)
        else:
            raise ValueError('splittype : holdout/kfold')

        return xtrain, xval, xpred, ytrain, yval, ypred, npred

    @TIMER.timer
    def unexisted_split(self, splittype='holdout', splits=0.2):
        '''
        目的・引数・返り値はunused_splitと同様

        '''
        if splittype == 'holdout':
            xtrain, xval, ytrain, yval = self.__unexisted__holdout(
                    self.X, self.Y, splits=splits)
            _, _, xpred, _, _, ypred, npred = self.__unused_holdout(
                    self.pred_x, self.pred_y, self.pred_n, splits=splits)
        elif splittype == 'kfold':
            xtrain, xval, ytrain, yval = self.__unexisted__kfold_cv(
                    self.X, self.Y, splits=splits)
            _, _, xpred, _, _, ypred, npred = self.__unused_kfold_cv(
                    self.pred_x, self.pred_y, self.pred_n, splits=splits)
        else:
            raise ValueError('splittype : holdout/kfold')

        return xtrain, xval, xpred, ytrain, yval, ypred, npred

    def __data_loader(self, datasetdir):
        '''
        目的：データセットディレクトリのクラスディレクトリにある画像を行列化・正解ベクトルの作成等
        引数：
            datasetdir : データセットディレクトリパス

        返り値：

            modeがRGBのとき, 画像行列・正解ベクトル・画像の名称
            modeがLのとき, channel次元を拡張した画像行列・正解ベクトル・画像の名称

            画像は最大画素値で割って正規化, 単精度小数に変換済み
            正解ベクトルはクラスディレクトリの数を検出して自動でそのクラス数にOneHot化する
            画像の名称はリストに文字列を格納し, タイプを文字列に変換するとベクトルにすることが可能

        '''
        if datasetdir is not None:
            class_list = os.listdir(datasetdir)

            xlist = list()
            ylist = list()
            nlist = list()
            for c in class_list:
                for img in glob.glob(os.path.join(datasetdir,c,'*.jpg')):
                    imgname = os.path.basename(img).split('.jpg')[0]

                    img = Image.open(img).convert(self.mode).resize(self.datashape[:2])
                    img = np.array(img)

                    xlist.append(img)
                    ylist.append(int(c))
                    nlist.append(imgname)

            if self.mode == 'L':
                x_matrix = np.array(xlist)[:,:,:,np.newaxis].astype('float32')/255.
            elif self.mode == 'RGB':
                x_matrix = np.array(xlist).astype('float32')/255.

            y_vector = to_categorical(np.array(ylist).astype('int32'), len(class_list))
            n_vector = np.array(nlist).astype('str')

            return x_matrix, y_vector, n_vector

    def __unused_holdout(self, x, y, n, splits=0.2):
        '''
        目的：予測用データに未使用画像を使うためのholdout分割関数

        引数：

            x : 画像データセット
            y : 正解ベクトルデータセット
            n : 画像名データセット
            splits : テストデータの分割割合, floatで0から1の範囲

        返り値：
            xtrain, ytrain : 訓練用の画像データと正解クラスベクトル
            xval, yval : 検証用の画像データと正解クラスベクトル
            xpred, ypred : 予測用の画像データと正解クラスベクトル
            npred : 予測用の画像データの名称を格納した配列

        '''
        if isinstance(splits, float):
            assert 0 < splits < 1, 'splits is between 0 and 1.'
        else:
            raise TypeError('splits : 0-1, float')

        xlearn, xpred, ylearn, ypred, _, npred = train_test_split(
                x,y,n,test_size=splits)
        xtrain, xval, ytrain, yval = train_test_split(
                xlearn, ylearn, test_size=splits)

        return xtrain, xval, xpred, ytrain, yval, ypred, npred

    def __unexisted__holdout(self, x, y, splits=0.2):
        '''
        目的：予測用データに未知画像を使うためのholdout分割関数

        引数：

            x : 画像データセット
            y : 正解ベクトルデータセット
            splits : テストデータの分割割合, floatで0から1の範囲

        返り値：
            xtrain, ytrain : 訓練用の画像データと正解クラスベクトル
            xval, yval : 検証用の画像データと正解クラスベクトル

            予測用は別に作成するので全学習データを訓練と検証に使える

        '''
        if isinstance(splits, float):
            assert 0 < splits < 1, 'splits is between 0 and 1.'
        else:
            raise TypeError('splits : 0-1, float')

        xtrain, xval, ytrain, yval = train_test_split(x,y,test_size=splits)

        return xtrain, xval, ytrain, yval

    def __unused_kfold_cv(self, x, y, n, splits=5):
        if isinstance(splits, (float, int)):
            if type(splits) is float:
                splits = int(splits)
        else:
            raise TypeError('splits : integer')

        for learn, pred in KFold(n_splits=splits, shuffle=True).split(x):
            xlearn, xpred = x[learn], x[pred]
            ylearn, ypred = y[learn], y[pred]
            _, npred = n[learn], n[pred]

        for train, val in KFold(n_splits=splits, shuffle=True).split(xlearn):
            xtrain, xval = xlearn[train], xlearn[val]
            ytrain, yval = ylearn[train], ylearn[val]

        return xtrain, xval, xpred, ytrain, yval, ypred, npred

    def __unexisted__kfold_cv(self, x, y, splits=5):
        if isinstance(splits, (float, int)):
            if type(splits) is float:
                splits = int(splits)
        else:
            raise TypeError('splits : integer')

        for train, val in KFold(n_splits=splits, shuffle=True).split(x):
            xtrain, xval = x[train], x[val]
            ytrain, yval = y[train], y[val]

        return xtrain, xval, ytrain, yval
