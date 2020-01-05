from preprocessing.data_preprocessing import ClassifyPreprocessing
from preprocessing.output_preprocessing import MidVisPreprocessing
from models.pretrain import Pretrained
from models.finetuning import FineTuning
from models.non_finetuning import NonFineTuning
from ft_utils import Timer
import os
import sys

TIMER = Timer() # 時間を計測して LINEにその結果を通知する

# ---------- commandline arguments ----------

assert len(sys.argv) == 5,\
'''

ex. python act_classify_dl.py pretrain original unused 0

argv 0 : python script file path
argv 1 : DL classify task name, pretrain/finetuning/normal-classify
argv 2 : model type to use in this classification, original/vgg16/vgg19
argv 3 : predicted data type, unused/unexisted
argv 4 : whether or not visualize DL model middle features, 0 is not, 1 is yes.

'''

DL_TYPE = sys.argv[1]
MODELNAME = sys.argv[2]
PRED_DATATYPE = sys.argv[3]
VIS_FLAG = int(sys.argv[4])

if DL_TYPE == 'pretrain':
    L_DATA = '1000_tkt'
    P_DATA = '0025'
elif DL_TYPE == 'finetuning':
    L_DATA = '1000_m-ito'
    P_DATA = '6712'
elif DL_TYPE == 'normal-classify':
    L_DATA = '1000_m-ito'
    P_DATA = '6712'
else:
    raise ValueError('Unknown DL-type.')

if VIS_FLAG == 0:
    VIS_FLAG = False
elif VIS_FLAG == 1:
    VIS_FLAG = True
else:
    raise ValueError('vis-flag is binary.')

# ---------- datasets and results dir setup ----------

RESULTS_HOME = os.path.join(
        os.path.expanduser('~'),
        'research_results/ml/finetuning'
        )
DATASETS_HOME = os.path.join(
        os.path.expanduser('~'),
        'work/git_work/datasets/supervised'
        )

LEARNING_DIR = os.path.join(
        DATASETS_HOME, 'learning', L_DATA
        )
PREDICTION_DIR = os.path.join(
        DATASETS_HOME, 'predict', P_DATA
        )
WEIGHT_SAVE_DIR = os.path.join(
        RESULTS_HOME, 'pretrain_weights'
        )
RESULTS_DIR = os.path.join(
        RESULTS_HOME, '{}-{}-{}'.format(
            DL_TYPE, L_DATA, P_DATA
            )
        )

# ---------- DL parameters ----------

HEIGHT = 56
WIDTH = 56
COLOR_MODE = 'RGB'
SPLITTYPE = 'holdout'

if SPLITTYPE == 'holdout':
    SPLITS = 0.2
elif SPLITTYPE == 'kfold':
    SPLITS = 5
else:
    raise Exception('Unknown')

GPUSAVE = False
SUMMARY = False

OPT = 'adam'
LEARNING_RATE = 5e-5
BETA_1 = 1e-6
BETA_2 = 0.9
MOMENTUM = 0.8
DECAY = 1e-5
FC_ACTIVATION = 'tanh'
ALPHA = 0.2
THETA = 1.

EPOCHS = 100
BATCH_SIZE = 20
VERBOSE = 0
CAPTIONS = True

OBJ = 'wing'
LAYER_NAME = 'MaxPooling2D'

# ---------- activated functions ----------

@TIMER.timer
def test_pretrain():
    # データセットの前処理クラスインスタンス化
    data = ClassifyPreprocessing(
            learndir=LEARNING_DIR, preddir=PREDICTION_DIR,
            height=HEIGHT, width=WIDTH, mode=COLOR_MODE
            )

    if PRED_DATATYPE == 'unused':
        # 予測用データ：未使用画像
        xtrain, xval, xpred, ytrain, yval, ypred, npred = data.unused_split(
                splittype=SPLITTYPE, splits=SPLITS
                )
    elif PRED_DATATYPE == 'unexisted':
        # 予測用データ：未知画像
        xtrain, xval, xpred, ytrain, yval, ypred, npred = data.unexisted_split(
                splittype=SPLITTYPE, splits=SPLITS
                )
    else:
        raise ValueError('pred-type : unused/unexisted')

    # 事前学習モデルのクラスインスタンス化
    pretrain = Pretrained(
            Xtrain=xtrain, Ytrain=ytrain, gpusave=GPUSAVE,
            summary=SUMMARY, ef_savedir=WEIGHT_SAVE_DIR,
            modelname=MODELNAME, resultdir=RESULTS_DIR
            )

    # 事前学習の実施
    pretrain.train(
            Xval=xval, Yval=yval, verbose=VERBOSE,
            captions=CAPTIONS, epochs=EPOCHS, batch_size=BATCH_SIZE
            )

    ## 念のため予測もやってみる
    pretrain.predict(Xpred=xpred, Ypred=ypred, Npred=npred, verbose=VERBOSE, objtype=OBJ)

    if VIS_FLAG:
        # 内部特徴量の可視化
        mid_vis = MidVisPreprocessing()
        #img_info = mid_vis.sample_preprocess(resize=(HEIGHT,WIDTH)) # 少数のサンプル画像を使用した中間層の可視化
        img_info = mid_vis.preprocessed_shape_convert(xpred, npred) # xpredの画像を使用した中間層の可視化
        pretrain.feature_intensity_vis(img_info_dict=img_info, layer_name=LAYER_NAME)

@TIMER.timer
def test_finetuning():
    ft_data = ClassifyPreprocessing(
            learndir=LEARNING_DIR, preddir=PREDICTION_DIR,
            height=HEIGHT, width=WIDTH, mode=COLOR_MODE
            )

    if PRED_DATATYPE == 'unused':
        xtrain, xval, xpred, ytrain, yval, ypred, npred = ft_data.unused_split(
                splittype=SPLITTYPE, splits=SPLITS)
    elif PRED_DATATYPE == 'unexisted':
        xtrain, xval, xpred, ytrain, yval, ypred, npred = ft_data.unexisted_split(
                splittype=SPLITTYPE, splits=SPLITS)
    else:
        raise ValueError('pred-type : unused/unexisted')

    # FineTuningモデルインスタンス
    finetuning = FineTuning(
            Xtrain=xtrain, Ytrain=ytrain, gpusave=GPUSAVE,
            summary=SUMMARY, modelname=MODELNAME, resultdir=RESULTS_DIR,
            weightsdir=WEIGHT_SAVE_DIR, fc_act=FC_ACTIVATION, alpha=ALPHA,
            theta=THETA
            )

    finetuning.train(
            xval, yval, verbose=VERBOSE,
            captions=CAPTIONS, epochs=EPOCHS, batch_size=BATCH_SIZE
            )
    finetuning.predict(xpred, ypred, npred, verbose=VERBOSE)

    if VIS_FLAG:
        mid_vis = MidVisPreprocessing()
        #img_info = mid_vis.sample_preprocess(resize=(HEIGHT,WIDTH))
        img_info = mid_vis.preprocessed_shape_convert(xpred, npred)
        finetuning.feature_intensity_vis(img_info_dict=img_info, layer_name=LAYER_NAME)

@TIMER.timer
def test_non_ft_classify():
    data = ClassifyPreprocessing(
            learndir=LEARNING_DIR, preddir=PREDICTION_DIR,
            height=HEIGHT, width=WIDTH, mode=COLOR_MODE
            )

    if PRED_DATATYPE == 'unused':
        xtrain, xval, xpred, ytrain, yval, ypred, npred = data.unused_split(
                splittype=SPLITTYPE, splits=SPLITS)
    elif PRED_DATATYPE == 'unexisted':
        xtrain, xval, xpred, ytrain, yval, ypred, npred = data.unexisted_split(
                splittype=SPLITTYPE, splits=SPLITS)
    else:
        raise ValueError('pred-type : unused/unexisted')

    non_ft = NonFineTuning(
            Xtrain=xtrain, Ytrain=ytrain, gpusave=GPUSAVE, summary=SUMMARY,
            resultdir=RESULTS_DIR
            )

    non_ft.train(
            Xval=xval, Yval=yval, verbose=VERBOSE,
            captions=CAPTIONS, epochs=EPOCHS, batch_size=BATCH_SIZE
            )
    non_ft.predict(Xpred=xpred, Ypred=ypred, Npred=npred, verbose=VERBOSE, objtype=OBJ)

    if VIS_FLAG:
        mid_vis = MidVisPreprocessing()
        #img_info = mid_vis.sample_preprocess(resize=(HEIGHT,WIDTH))
        img_info = mid_vis.preprocessed_shape_convert(xpred, npred)
        non_ft.feature_intensity_vis(img_info_dict=img_info, layer_name=LAYER_NAME)

if __name__ == '__main__':
    if DL_TYPE == 'pretrain':
        test_pretrain()
    elif DL_TYPE == 'finetuning':
        test_finetuning()
    elif DL_TYPE == 'normal-classify':
        test_non_ft_classify()
    else:
        raise ValueError('Unknown DL type.')
