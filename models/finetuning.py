from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import ELU
from keras.layers.advanced_activations import ThresholdedReLU
from keras.optimizers import Adam, Adagrad, Adamax, Adadelta, SGD
from keras.optimizers import Nadam
from keras.callbacks import CSVLogger
from keras import backend as K
from ft_utils import mkdirs, Timer
from ft_utils import KerasVisualizer
from models.indicators import LearningIndicators
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob

TIMER = Timer()

if os.name == 'posix':
    NEWLINECODE = '\n'
elif os.name == 'nt':
    NEWLINECODE = '\r\n'
else:
    raise OSError('Unknown OS.')

class FineTuning(object):
    def __init__(self, Xtrain, Ytrain, gpusave=False, summary=True, summaryout=False,
            weightsdir=None, opt='adam', lr=5e-5, beta_1=1e-6, beta_2=0.9,
            amsgrad=True, momentu=0.8, decay=1e-5, nesterov=False, decay_on=False,
            modelname='original', resultdir=None, fc_act='relu', alpha=0.3, theta=1.):
        if gpusave:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            K.set_session(tf.Session(config=config))

        assert weightsdir is not None,\
                'Input pretrain model weights directory.'

        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.summary = summary
        self.summaryout = summaryout
        self.modelname = modelname

        self.activation = fc_act
        self.alpha = alpha
        self.theta = theta

        weight = [w for w in glob.glob(os.path.join(weightsdir,'*')) if self.modelname in w][0]
        print(weight)
        _, self.model = self.__load_pretrain(weight)

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

        self.model.compile(loss=loss, optimizer=optimizer, metrics=met)

        if resultdir is not None:
            self.vis = KerasVisualizer(savedir=resultdir)
        else:
            self.vis = KerasVisualizer(savedir='./ft_results')

    @TIMER.timer
    def train(self, Xval, Yval, epochs=100, batch_size=20, verbose=1, captions=True):
        csv = CSVLogger(os.path.join(
            self.vis.savedir,'{}.csv'.format(self.model.name)))

        history = self.model.fit(self.Xtrain, self.Ytrain, epochs=epochs,
                batch_size=batch_size, verbose=verbose,
                validation_data=(Xval,Yval), callbacks=[csv])

        for indicator in [indicator for indicator in history.history.keys() if not 'val_' in indicator]:
            self.vis.keras_histsave(history, indicator, showinfo=captions)

    @TIMER.timer
    def predict(self, Xpred, Ypred, Npred, batch_size=None,
            verbose=1, objtype='wing', norm=0.3, saveext=None):
        assert Ypred.shape[-1] == 5,\
                'class number : 5'

        class_probs = self.model.predict(Xpred,
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
            'p-3':[class_probs[i,3] for i in range(class_probs.shape[0])],
            'p-4':[class_probs[i,4] for i in range(class_probs.shape[0])]
            }).sort_values('name')
        df.index = range(class_probs.shape[0])

        transitions = list()
        for i in range(Ypred.shape[0]):
            transition = '----'

            if df.loc[i,'p-0']>norm and df.loc[i,'p-2']>norm and df.loc[i,'p-0']>df.loc[i,'p-2']:
                transition = '0 -> 2'
            elif df.loc[i,'p-2']>norm and df.loc[i,'p-1']>norm and df.loc[i,'p-2']>df.loc[i,'p-1']:
                transition = '2 -> 1'
            elif df.loc[i,'p-2']>norm and df.loc[i,'p-4']>norm and df.loc[i,'p-2']>df.loc[i,'p-4']:
                transition = '2 -> 4'
            elif df.loc[i,'p-4']>norm and df.loc[i,'p-1']>norm and df.loc[i,'p-4']>df.loc[i,'p-1']:
                transition = '4 -> 1'
            elif df.loc[i,'p-1']>norm and df.loc[i,'p-3']>norm and df.loc[i,'p-1']>df.loc[i,'p-3']:
                transition = '1 -> 3'

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
        for layer in self.model.layers:
            layer_type = layer.__class__
            name = layer_type.__name__

            layer_types[name] = layer_type

        if img_info_dict is not None:
            if not isinstance(img_info_dict,dict):
                raise TypeError('img_info_dict : dict')

        assert layer_name in list(layer_types.keys()),\
                'Not found layer object type..'

        layers = self.model.layers[1:-1] # 入力層と出力層以外の層
        layer_outputs = [layer.output for layer in layers]

        mid_model = Model(inputs=self.model.input, outputs=layer_outputs)
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

    def __load_pretrain(self, loadmodel):
        pretrain = load_model(loadmodel)

        inputs = Input(shape=pretrain.output_shape[1:])
        fc = Flatten()(inputs)
        fc = self.__activation_arrange(fc,
                activation=self.activation,
                alpha=self.alpha, theta=self.theta)
        fc = Dropout(0.1)(fc)
        outputs = Dense(self.Ytrain.shape[-1], activation='softmax')(fc)

        top_model = Model(inputs, outputs)
        total_model = Model(pretrain.inputs, top_model(pretrain.outputs),
                name='{}_finetuning'.format(self.modelname))

        if self.modelname == 'original':
            fix_layers = total_model.layers[:5]
        elif self.modelname == 'vgg16':
            fix_layers = total_model.layers[:15]
        elif self.modelname == 'vgg19':
            fix_layers = total_model.layers[:17]
        else:
            raise ValueError('modelname : original/vgg16/vgg19')

        for layer in fix_layers:
            layer.trainable = False

        if self.summary:
            self.__summary(top_model, self.summaryout)
            self.__summary(total_model, self.summaryout)

        return top_model, total_model

    def __activation_arrange(self, input_tensor, activation='relu', alpha=0.3, theta=1.):
        default_activations = (
                'relu','tanh','softplus','softsign',
                'elu','selu','sigmoid','hard_sigmoid',
                'linear'
                )
        advanced_activations = (
                'leakyrelu','prelu','elu','threshold'
                )

        if activation in default_activations:
            x = Dense(1024, activation=activation)(input_tensor)
        elif activation in advanced_activations:
            x = Dense(1024)(input_tensor)

            if activation == 'leakyrelu':
                assert 0<alpha<1, 'Incorrect alpha.'
                x = LeakyReLU(alpha)(x)
            elif activation == 'prelu':
                x = PReLU()(x)
            elif activation == 'elu':
                x = ELU(alpha)(x)
            elif activation == 'threshold':
                assert theta>=0, 'Incorrect threshold-values.'
                x = ThresholdedReLU(theta)(x)

        else:
            raise ValueError('Not Found Activation Functions.')

        return x

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
