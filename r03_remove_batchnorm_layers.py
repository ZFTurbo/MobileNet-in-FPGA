# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

# Remove layers which is not needed for inference using KITO script

import os
import glob

if __name__ == '__main__':
    # Block to choose GPU
    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from kito import reduce_keras_model
from keras.models import load_model
from a00_common_functions import *


if __name__ == '__main__':
    model_path_in = MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33.h5'
    model_path_out = MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33_bnfused.h5'
    model = load_model(model_path_in, custom_objects={'relu_1': relu_1})
    model_reduced = reduce_keras_model(model, verbose=True)
    print(model_reduced.summary())
    print('Initial layers: {}'.format(len(model.layers)))
    print('Reduced layers: {}'.format(len(model_reduced.layers)))
    model_reduced.save(model_path_out)


'''
MobileNet V1 (Keras 2.2.4)
Initial layers: 89
Reduced layers: 62
'''