# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

# Train MobileNet with batch generator and augmentations
# Made for training with tensorflow only

import os
import glob

if __name__ == '__main__':
    # Block to choose GPU
    gpu_use = 0
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


MOBILENET_VERSION = 1
MOBILENET_ALFA = 1.0
MOBILENET_INPUT_SIZE = 128


from functools import partial
from keras import backend as K
from keras.optimizers import SGD, Adam
if MOBILENET_VERSION == 1:
    from keras.applications.mobilenet import MobileNet, preprocess_input
else:
    from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.layers.core import Dense
from keras.models import Model
from a01_oid_utils import *
from a00_common_functions import *
from albumentations import *
import pandas as pd
import pyvips
from PIL import Image
from r01_prepare_open_images_dataset import DATASET_PATH
from multiprocessing.pool import ThreadPool
import random


def strong_aug(p=.5):
    return Compose([
        # RandomRotate90(),
        HorizontalFlip(p=0.5),
        # Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.1),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.1),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.1),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.1),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
        ], p=0.3),
        RGBShift(p=0.1, r_shift_limit=(-30, 30), g_shift_limit=(-30, 30), b_shift_limit=(-30, 30)),
        RandomBrightnessContrast(p=0.05),
        HueSaturationValue(p=0.05),
        ToGray(p=0.05),
        JpegCompression(p=0.05, quality_lower=55, quality_upper=99),
        ElasticTransform(p=0.05),
    ], p=p)


GLOBAL_AUG = strong_aug(p=1.0)


def process_single_item(id, box_size, validation=True):
    global global_aug

    if validation is not True:
        img = read_image_bgr_fast(DATASET_PATH + 'train/' + id[:3] + '/' + id + '.jpg')
    else:
        img = read_image_bgr_fast(DATASET_PATH + 'validation_big/' + id + '.jpg')
    if img is None:
        img = np.zeros((box_size, box_size, 3), dtype=np.uint8)

    if validation is not True:
        img = GLOBAL_AUG(image=img)['image']
        if 0:
            img = random_intensity_change(img, 10)
            img = random_rotate(img, 10)
            if random.randint(0, 1) == 0:
                # fliplr
                img = img[:, ::-1, :]
    if img.shape[0] != box_size:
        img = cv2.resize(img, (box_size, box_size), cv2.INTER_LANCZOS4)

    return img


def batch_generator(X_train, Y_train, batch_size, input_size, prep_input, validation):
    threads = 8
    p = ThreadPool(threads)
    process_item_func = partial(process_single_item, validation=validation, box_size=input_size)

    while True:
        batch_indexes = np.random.choice(X_train.shape[0], batch_size)
        batch_image_files = X_train[batch_indexes].copy()
        batch_classes = Y_train[batch_indexes].copy()
        batch_images = p.map(process_item_func, batch_image_files)
        batch_images = np.array(batch_images, np.float32)
        batch_images = prep_input(batch_images)
        yield batch_images.copy(), batch_classes


def evaluate_generator(X_test, Y_test, batch_size, input_size, prep_input):
    number_of_batches = X_test.shape[0] // batch_size
    target_size = input_size

    i = 0
    while 1:
        batch_images = np.zeros((batch_size, target_size, target_size, 3))
        if i >= number_of_batches:
            print('Current {}'.format(i))
            batch_image_files = X_test[-batch_size:]
            batch_classes = Y_test[-batch_size:]
        else:
            batch_image_files = X_test[i*batch_size:(i+1)*batch_size]
            batch_classes = Y_test[i*batch_size:(i+1)*batch_size]

        # Rescale to 128x128
        for j in range(batch_size):
            img = cv2.imread(batch_image_files[j])
            if img.shape[0] != target_size:
                img = cv2.resize(img, (target_size, target_size), cv2.INTER_LANCZOS4)
            batch_images[j, :, :, :] = img
        batch_images = prep_input(batch_images)
        i += 1
        yield batch_images, batch_classes


def load_train_valid_data(train_csv, valid_csv):
    from keras.utils import to_categorical
    valid = pd.read_csv(train_csv)
    train = pd.read_csv(valid_csv)
    X_train = train['id'].values
    Y_train = to_categorical(train['target'].values, num_classes=2)
    X_valid = valid['id'].values
    Y_valid = to_categorical(valid['target'].values, num_classes=2)
    return  X_train, Y_train, X_valid, Y_valid


def train_mobile_net_v1(input_size, train_csv, valid_csv):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
    batch_size = 32
    nb_classes = 2
    nb_epoch = 1000
    patience = 50
    optimizer = 'SGD'
    learning_rate = 0.00005
    restore = 1

    print('Train MobileNet version: {} Input size: {}'.format(MOBILENET_VERSION, input_size))
    print('Train for {} epochs with patience {}. Batch size: {}. Optimizer: {} Learing rate: {}'.
          format(nb_epoch, patience, batch_size, optimizer, learning_rate))

    X_train, Y_train, X_valid, Y_valid = load_train_valid_data(train_csv, valid_csv)
    print('Train shape: {}'.format(X_train.shape))
    print('Valid shape: {}'.format(X_valid.shape))

    print('Dim ordering:', K.image_dim_ordering())
    if MOBILENET_VERSION == 1:
        alpha = MOBILENET_ALFA
        base_model = MobileNet((input_size, input_size, 3), depth_multiplier=1, alpha=alpha,
                               include_top=False, pooling='avg', weights='imagenet')
    else:
        alpha = 0.35
        base_model = MobileNetV2((input_size, input_size, 3), depth_multiplier=1, alpha=alpha,
                                 include_top=False, pooling='avg', weights='imagenet')
    x = base_model.output
    x = Dense(nb_classes, activation='softmax', name='predictions', use_bias=False)(x)
    model = Model(input=base_model.input, output=x)
    print(model.summary())

    if optimizer == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model memory usage: {:.3f} GB'.format(get_model_memory_usage(batch_size, model)))

    if not os.path.isdir('cache'):
        os.mkdir('cache')
    cache_model_path = os.path.join(MODEL_PATH, 'weights_mobilenet_{}_{:.2f}_{}px_people_v2.h5'.format(MOBILENET_VERSION, alpha, input_size))
    if os.path.isfile(cache_model_path) and restore:
        print('Restore weights from cache: {}'.format(cache_model_path))
        model.load_weights(cache_model_path)

    history_path = os.path.join(MODEL_PATH,
                                'weights_mobilenet_{}_{:.2f}_{}px_people_v2.csv'.format(MOBILENET_VERSION, alpha,
                                                                                        input_size))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        CSVLogger(MODEL_PATH + 'history_people_lr_{}_optim_{}_v2.csv'.format(learning_rate, optimizer), append=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=1e-9, min_delta=0.00001, verbose=1,
                          mode='min'),
    ]

    steps_per_epoch = 1000
    validation_steps = X_valid.shape[0] // batch_size
    history = model.fit_generator(generator=batch_generator(X_train, Y_train, batch_size, input_size, preprocess_input, validation=False),
                                  epochs=nb_epoch,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=batch_generator(X_valid, Y_valid, batch_size, input_size, preprocess_input, validation=True),
                                  validation_steps=validation_steps,
                                  verbose=2,
                                  max_queue_size=20,
                                  callbacks=callbacks)
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    score = model.evaluate_generator(generator=evaluate_generator(X_valid, Y_valid, batch_size, input_size, preprocess_input),
                                     steps=X_valid.shape[0] // batch_size,
                                     max_queue_size=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print('Best model stored in {}'.format(cache_model_path))


if __name__ == '__main__':
    input_size = MOBILENET_INPUT_SIZE
    train_csv = CACHE_PATH + 'oid_train_animals.csv'
    valid_csv = CACHE_PATH + 'oid_validation_animals.csv'
    train_mobile_net_v1(input_size, train_csv, valid_csv)



'''
Animals MobileNet v1 (0.25, 128px): 

'''