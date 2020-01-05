from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import Adam, Adagrad, Adamax, Adadelta, SGD
from keras.optimizers import Nadam
from keras.callbacks import CSVLogger
from keras.preprocessing import image
from keras import backend as K
from ft_utils import mkdirs, Timer
from ft_utils import KerasVisualizer
from models.indicators import LearningIndicators
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import inspect # 引数の情報を確認できる

TIMER = Timer()

if os.name == 'posix':
    NEWLINECODE = '\n'
elif os.name == 'nt':
    NEWLINECODE = '\r\n'
else:
    raise OSError('Unknown OS.')

class Pretrained(object):
    def __init__(self, Xtrain, Ytrain, gpusave=False, summary=True, summaryout=False,
            ef_savedir=None, opt='adam', lr=5e-5, beta_1=1e-6, beta_2=0.9,
            amsgrad=True, momentu=0.8, decay=1e-5, nesterov=False, decay_on=False,
            modelname='original', resultdir=None):
        '''
        目的：事前学習用のモデル定義と最適化, 特徴抽出層の保存

        引数：

            Xtrain : 訓練画像データ行列
            Ytrain : 訓練正解クラスベクトル
            gpusave : 学習時のGPU使用率を節約するフラッグ, デフォルトはFalse
            summary : モデルパラメータ－を表示するフラッグ, デフォルトはTrue
            summaryout : summaryの結果をテキストファイルにするフラッグ, デフォルトはFalse
            ef_savedir : 特徴抽出モデル層を重み付きで保存するためのディレクトリパス, Noneだと保存しない
            opt : 学習で使用する最適化アルゴリズム, デフォルトはadam

                adam : Adam
                adagrad : Adagrad
                adamax : Adamax
                adadelta : Adadelta
                sgd : SGD

            lr : 学習率
            beta_1 : Adam等で使用する正則化パラメーター
            beta_2 : Adam等で使用する正則化パラメーター
            amsgrad : Adam等で使用するアルゴリズム変更フラッグ, デフォルトはTrue
            momentum : SGD等で使用する正則化パラメーター
            decay : 勾配更新で収束を促すパラメーター
            nesterov : SGDで使用するアルゴリズム変更フラッグ, デフォルトはFalse
            decay_on : Adam等でもdecayを設定するためのフラッグ, デフォルトはFalse
            modelname : 事前学習モデル

                original : 伊藤のオリジナルモデル
                vgg16 : VGG16
                vgg19 : VGG19

                ※ 全結合層は全てoriginalと同様

            resultdir : 学習曲線等を保存するためのディレクトリパス, Noneのときはカレントに自動作成

        '''
        if gpusave:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            K.set_session(tf.Session(config=config))

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.summary = summary
        self.summaryout = summaryout

        if modelname == 'original':
            self.ef_model, self.pretrained = self.__original_forward()
        elif modelname == 'vgg16':
            self.ef_model, self.pretrained = self.__vgg16_forward()
        elif modelname == 'vgg19':
            self.ef_model, self.pretrained = self.__vgg19_forward()
        else:
            raise ValueError('Not found extracting features model.')

        if opt == 'adam':
            if decay_on:
                optimizer = Adam(lr=lr,beta_1=beta_1,beta_2=beta_2,
                        amsgrad=amsgrad,decay=decay)
            else:
                optimizer = Adam(lr=lr,beta_1=beta_1,beta_2=beta_2,amsgrad=amsgrad)
        elif opt == 'adamax':
            if decay_on:
                optimizer = Adamax(lr=lr,beta_1=beta_1,beta_2=beta_2,
                        amsgrad=amsgrad,decay=decay)
            else:
                optimizer = Adamax(lr=lr,beta_1=beta_1,beta_2=beta_2,amsgrad=amsgrad)
        elif opt == 'adagrad':
            optimizer = Adagrad()
        elif opt == 'adadelta':
            optimizer = Adadelta()
        elif opt == 'sgd':
            optimizer = SGD(lr=lr,momentum=momentum,decay=decay,nesterov=nesterov)
        elif opt == 'nadam':
            optimizer = Nadam(lr=lr,beta_1=beta_1,beta_2=beta_2,schedule_decay=decay)

        loss = 'categorical_crossentropy'
        met = LearningIndicators(modeltype='classify',
                num_classes=self.Ytrain.shape[-1]).metrics

        self.ef_model.compile(loss=loss, optimizer=optimizer)
        self.pretrained.compile(loss=loss, optimizer=optimizer, metrics=met)

        if ef_savedir is not None:
            ef_savedir = mkdirs(ef_savedir)
            self.ef_model.save(os.path.join(
                ef_savedir,'{}_efmodel.h5'.format(modelname)))

        if resultdir is not None:
            self.vis = KerasVisualizer(savedir=resultdir)
        else:
            self.vis = KerasVisualizer(savedir='./learning_results')

    @TIMER.timer
    def train(self, Xval, Yval, epochs=100, batch_size=20, verbose=1, captions=True):
        csv = CSVLogger(os.path.join(
            self.vis.savedir,'{}.csv'.format(self.pretrained.name)))

        history = self.pretrained.fit(self.Xtrain, self.Ytrain, epochs=epochs,
                batch_size=batch_size, verbose=verbose,
                validation_data=(Xval,Yval), callbacks=[csv])

        for indicator in [indicator for indicator in history.history.keys() if not 'val_' in indicator]:
            self.vis.keras_histsave(history, indicator, showinfo=captions)

    @TIMER.timer
    def predict(self, Xpred, Ypred, Npred, batch_size=None,
            verbose=1, objtype='wing', norm=0.3, saveext=None):
        assert Ypred.shape[-1] == 4,\
                'class number : 4'

        class_probs = self.pretrained.predict(Xpred,
                verbose=verbose, batch_size=batch_size)

        trues = [np.argmax(Ypred[i]) for i in range(Ypred.shape[0])]
        preds = [np.argmax(class_probs[i]) for i in range(class_probs.shape[0])]
        match = [0 for true, pred in zip(trues, preds) if true == pred]

        with open(os.path.join(self.vis.savedir,'matching.out'), 'w') as p_result:
            p_result.write('matching rate : {:.3f}'.format(len(match)/len(trues)*100))

        df = pd.DataFrame({
            'name':Npred,
            'true':trues,
            'pred':preds,
            'p-0':[class_probs[i,0] for i in range(class_probs.shape[0])],
            'p-1':[class_probs[i,1] for i in range(class_probs.shape[0])],
            'p-2':[class_probs[i,2] for i in range(class_probs.shape[0])],
            'p-3':[class_probs[i,3] for i in range(class_probs.shape[0])]
            }).sort_values('name')
        df.index = range(class_probs.shape[0])

        transitions = list()
        for i in range(Ypred.shape[0]):
            transition = '----'

            # loc[行名, 列名]で値を取得可能, インデックスでの指定はilocで可能
            # デフォルトの行名は整数値, 0から始まる
            if df.loc[i,'p-0']>norm and df.loc[i,'p-1']>norm and df.loc[i,'p-0']>df.loc[i,'p-1']:
                transition = '0 -> 1'
            elif df.loc[i,'p-0']>norm and df.loc[i,'p-2']>norm and df.loc[i,'p-0']>df.loc[i,'p-2']:
                transition = '0 -> 2'
            elif df.loc[i,'p-1']>norm and df.loc[i,'p-3']>norm and df.loc[i,'p-1']>df.loc[i,'p-3']:
                transition = '1 -> 3'
            elif df.loc[i,'p-2']>norm and df.loc[i,'p-3']>norm and df.loc[i,'p-2']>df.loc[i,'p-3']:
                transition = '2 -> 3'

            transitions.append(transition)

        df['transition'] = transitions
        df.dtype = float

        if saveext is not None:
            if 'csv' in saveext:
                df.to_csv(os.path.join(self.vis.savedir,'transitions.csv'))
            elif 'xlsx' in saveext:
                df.to_excel(os.path.join(self.vis.savedir,'transitions.xlsx'))
            elif 'html' in savefile:
                df.to_html(os.path.join(self.vis.savedir,'transitions.html'))
            else:
                raise ValueError('Not converted into file.')
        else:
            df.to_csv(os.path.join(self.vis.savedir,'transitions.csv'))

    @TIMER.timer
    def feature_intensity_vis(self, img_info_dict=None, layer_name='MaxPooling2D'):
        layer_types = {}
        for layer in self.pretrained.layers:
            layer_type = layer.__class__
            name = layer_type.__name__

            layer_types[name] = layer_type

        if img_info_dict is not None:
            if not isinstance(img_info_dict,dict):
                raise TypeError('img_info_dict : dict')

        assert layer_name in list(layer_types.keys()),\
                'Not found layer object type..'

        layers = self.pretrained.layers[1:-1] # 入力層と出力層以外の層
        layer_outputs = [layer.output for layer in layers]

        mid_model = Model(inputs=self.pretrained.input, outputs=layer_outputs)
        if self.summary:
            self.__summary(mid_model, self.summaryout)

        for key in img_info_dict.keys():
            img_mat = img_info_dict[key]

            activations = mid_model.predict(img_mat)
            # MaxPooling2Dの特徴量のみを取り出す
            # MaxPooling2Dで圧縮した後の特徴強度を可視化する
            activations = [
                    (layer.name,activation) for layer,activation in zip(layers,activations)\
                    if isinstance(layer,layer_types[layer_name])
                    ]

            self.vis.middle_layer_visualizer(model_matrix=activations, img_name=key)

    def __original_forward(self):
        '''
        目的：伊藤オリジナルモデルのforward
        引数：なし
        返り値：特徴抽出モデル, 全体モデル

        '''
        inputs = Input(shape=self.Xtrain.shape[1:])

        hid = Conv2D(16,(5,5),padding='same',activation='relu')(inputs)
        hid = Conv2D(32,(5,5),padding='same',activation='relu')(hid)
        hid = MaxPooling2D((2,2))(hid)
        hid = Dropout(0.3)(hid)
        hid = Conv2D(64,(5,5),padding='same',activation='relu')(hid)
        hid = Conv2D(64,(5,5),padding='same',activation='relu')(hid)
        hid = MaxPooling2D((2,2))(hid)
        hid = Dropout(0.25)(hid)

        fc = Flatten()(hid)
        fc = Dense(1024,activation='relu')(fc)
        fc = Dropout(0.1)(fc)

        outputs = Dense(self.Ytrain.shape[-1],activation='softmax')(fc)

        ef_model = Model(inputs, hid, name='extracting-model')
        pretrained_model = Model(inputs, outputs, name='pretrained-model')
        if self.summary:
            self.__summary(ef_model, self.summaryout)
            self.__summary(pretrained_model, self.summaryout)

        return ef_model, pretrained_model

    def __vgg16_forward(self):
        '''
        originalと同様
        '''
        ef_model = VGG16(weights=None, include_top=False,
                input_shape=self.Xtrain.shape[1:])

        fc_inputs = Input(shape=ef_model.output_shape[1:])
        fc = Flatten()(fc_inputs)
        fc = Dense(1024,activation='relu')(fc)
        fc = Dropout(0.1)(fc)
        fc_outputs = Dense(self.Ytrain.shape[-1],activation='softmax')(fc)

        top_model = Model(fc_inputs, fc_outputs, name='VGG16-top')
        pretrained_model = Model(ef_model.inputs, top_model(ef_model.outputs),
                name='VGG16-pretrained')
        if self.summary:
            self.__summary(ef_model, self.summaryout)
            self.__summary(pretrained_model, self.summaryout)

        return ef_model, pretrained_model

    def __vgg19_forward(self):
        '''
        originalと同様
        '''
        ef_model = VGG19(weights=None, include_top=False,
                input_shape=self.Xtrain.shape[1:])

        fc_inputs = Input(shape=ef_model.output_shape[1:])
        fc = Flatten()(fc_inputs)
        fc = Dense(1024,activation='relu')(fc)
        fc = Dropout(0.1)(fc)
        fc_outputs = Dense(self.Ytrain.shape[-1],activation='softmax')(fc)

        top_model = Model(fc_inputs, fc_outputs, name='VGG19-top')
        pretrained_model = Model(ef_model.inputs, top_model(ef_model.outputs),
                name='VGG19-pretrained')
        if self.summary:
            self.__summary(ef_model, self.summaryout)
            self.__summary(pretrained_model, self.summaryout)

        return ef_model, pretrained_model

    def __summary(self, model, summaryout=False):
        '''
        目的：モデルパラメーターの表示

        引数：
            model : モデルクラスオブジェクト
            summaryout : summary結果をテキストファイルにするフラッグ, デフォルトはFalse

        '''
        if summaryout:
            with open('./{}_summary.txt'.format(model.name), 'w') as s:
                model.summary(print_ln=lambda x:s.write(x+NEWLINECODE))
        else:
            model.summary()
