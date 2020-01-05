from keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import glob

class MidVisPreprocessing(object):
    def __init__(self, filedir=None):
        if filedir is not None:
            self.imgs = glob.glob(os.path.join(filedir,'*.jpg'))

    def sample_preprocess(self, resize=(56,56)):
        if not isinstance(resize, tuple):
            raise TypeError('resize parameter : tuple type')

        img_info = {}
        for img in self.imgs:
            img_name = os.path.basename(img).split('.jpg')[0]

            img = image.load_img(img,target_size=self.resize)
            img = image.img_to_array(img)
            img = np.expand_dims(img,axis=0)/255.

            img_info[img_name] = img

        return img_info

    def preprocessed_shape_convert(self, preprocessed_tensor=None, img_names=None):
        if not isinstance(preprocessed_tensor, np.ndarray):
            raise TypeError('preprocessed_tensor : 4D-numpy.ndarray')

        if not isinstance(img_names, (list, np.ndarray)):
            raise TypeError('img_names : list or 1D-numpy.ndarray')

        img_info = {}
        for name, tensor in zip(img_names, preprocessed_tensor):
            tensor = tensor[np.newaxis,:,:,:]
            img_info[name] = tensor

        return img_info
