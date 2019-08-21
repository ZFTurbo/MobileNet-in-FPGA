# Util functions
import pickle
import gzip
import cv2
import numpy as np
import pandas as pd
import os
import glob
import random

ROOT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
MODEL_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def relu_1(x):
    from keras.activations import relu
    return relu(x, max_value=1.0)


def save_history(history, path, columns=('loss', 'val_loss')):
    import matplotlib.pyplot as plt
    import pandas as pd
    s = pd.DataFrame(history.history)
    s.to_csv(path + '.csv')
    plt.plot(s[list(columns)])
    plt.savefig(path + '.png')
    plt.close()


def get_model(weights_path):
    from keras.models import load_model
    print('Load: {}'.format(weights_path))
    model = load_model(weights_path, custom_objects={'relu_1': relu_1})
    print('Number of layers: {}'.format(len(model.layers)))
    return model


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes