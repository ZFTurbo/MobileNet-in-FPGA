# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

# Train MobileNet with batch generator and augmentations
# Made for training with tensorflow only

import os
import glob

if __name__ == '__main__':
    # Block to choose GPU
    gpu_use = 2
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)


MOBILENET_VERSION = 1
MOBILENET_ALFA = 0.25
MOBILENET_INPUT_SIZE = 128
CHANNEL_TYPE = 'RGB'


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
from r01_prepare_open_images_dataset import DATASET_PATH
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
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

    # Important: RGB order!
    if validation is not True:
        file_path = DATASET_PATH + 'train/' + id[:3] + '/' + id + '.jpg'
    else:
        file_path = DATASET_PATH + 'validation/' + id + '.jpg'
    if CHANNEL_TYPE == 'RGB':
        img = read_single_image(file_path)
    else:
        img = read_image_bgr_fast(file_path)
    if img is None:
        img = np.zeros((box_size, box_size, 3), dtype=np.uint8)

    if validation is not True:
        # img = GLOBAL_AUG(image=img)['image']
        if 1:
            img = random_intensity_change(img, 10)
            img = random_rotate(img, 10)
            if random.randint(0, 1) == 0:
                # fliplr
                img = img[:, ::-1, :]
    if img.shape[0] != box_size:
        img = cv2.resize(img, (box_size, box_size), interpolation=cv2.INTER_LINEAR)

    return img


def batch_generator(X_train, Y_train, batch_size, input_size, prep_input, validation):
    threads = cpu_count() - 1
    p = ThreadPool(threads)
    process_item_func = partial(process_single_item, validation=validation, box_size=input_size)

    # Do around 50% of batch to have required class
    X_train_no_class = X_train[Y_train[:, 1] == 0]
    X_train_with_class = X_train[Y_train[:, 1] == 1]
    Y_train_no_class = Y_train[Y_train[:, 1] == 0]
    Y_train_with_class = Y_train[Y_train[:, 1] == 1]
    print('Use threads: {}'.format(threads))
    print(X_train_no_class.shape, X_train_with_class.shape, Y_train_no_class.shape, Y_train_with_class.shape)
    b1 = batch_size // 2
    b2 = batch_size - b1

    while True:
        batch_indexes_no_cars = np.random.choice(X_train_no_class.shape[0], b1)
        batch_indexes_with_cars = np.random.choice(X_train_with_class.shape[0], b2)
        batch_image_files = np.concatenate(
            (X_train_no_class[batch_indexes_no_cars].copy(), X_train_with_class[batch_indexes_with_cars].copy())
        )
        batch_classes = np.concatenate(
            (Y_train_no_class[batch_indexes_no_cars].copy(), Y_train_with_class[batch_indexes_with_cars].copy())
        )
        batch_images = p.map(process_item_func, batch_image_files)
        batch_images = np.array(batch_images, np.float32)
        batch_images = prep_input(batch_images)
        yield batch_images, batch_classes


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
            path = DATASET_PATH + 'validation/' + batch_image_files[j] + '.jpg'
            if CHANNEL_TYPE == 'RGB':
                img = read_single_image(path)
            else:
                img = read_image_bgr_fast(path)
            if img.shape[0] != target_size:
                img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            batch_images[j, :, :, :] = img
        batch_images = prep_input(batch_images)
        i += 1
        yield batch_images, batch_classes


def load_train_valid_data(train_csv, valid_csv):
    from keras.utils import to_categorical
    valid = pd.read_csv(valid_csv)
    train = pd.read_csv(train_csv)
    X_train = train['id'].values
    Y_train = to_categorical(train['target'].values, num_classes=2)
    X_valid = valid['id'].values
    Y_valid = to_categorical(valid['target'].values, num_classes=2)
    return  X_train, Y_train, X_valid, Y_valid


def train_mobile_net_v1(input_size, train_csv, valid_csv, type):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
    batch_size = 1024
    nb_classes = 2
    nb_epoch = 1000
    patience = 50
    optimizer = 'Adam'
    learning_rate = 0.0001
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
    model = Model(inputs=base_model.input, outputs=x)
    print(model.summary())

    if optimizer == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Model memory usage: {:.3f} GB'.format(get_model_memory_usage(batch_size, model)))

    if not os.path.isdir('cache'):
        os.mkdir('cache')
    prefix = 'mobilenet_{}_{:.2f}_{}px_{}'.format(MOBILENET_VERSION, alpha, input_size, type)
    cache_model_path = os.path.join(MODEL_PATH, 'weights_{}.h5'.format(prefix))
    cache_model_path_score = MODEL_PATH + 'weights_{}_'.format(prefix) + 'loss_{val_loss:.4f}_acc_{val_acc:.4f}_epoch_{epoch:02d}_' + '{}.h5'.format(CHANNEL_TYPE)
    if os.path.isfile(cache_model_path) and restore:
        print('Restore weights from cache: {}'.format(cache_model_path))
        model.load_weights(cache_model_path)

    history_path = os.path.join(MODEL_PATH,
                                'weights_mobilenet_{}_{:.2f}_{}px_people_v2.csv'.format(MOBILENET_VERSION, alpha,
                                                                                        input_size))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        ModelCheckpoint(cache_model_path_score, monitor='val_loss', save_best_only=False, verbose=0),
        CSVLogger(MODEL_PATH + 'history_{}_lr_{}_optim_{}_v2.csv'.format(type, learning_rate, optimizer), append=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=1e-9, min_delta=0.00001, verbose=1, mode='min'),
    ]

    steps_per_epoch = 100
    validation_steps = X_valid.shape[0] // batch_size
    history = model.fit_generator(generator=batch_generator(X_train, Y_train, batch_size, input_size, preprocess_input, validation=False),
                                  epochs=nb_epoch,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=batch_generator(X_valid, Y_valid, batch_size, input_size, preprocess_input, validation=True),
                                  validation_steps=validation_steps,
                                  verbose=1,
                                  max_queue_size=16,
                                  initial_epoch=0,
                                  callbacks=callbacks)
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    score = model.evaluate_generator(generator=evaluate_generator(X_valid, Y_valid, batch_size, input_size, preprocess_input),
                                     steps=X_valid.shape[0] // batch_size,
                                     max_queue_size=1)
    print('Full validation loss: {:.4f} Full validation accuracy: {:.4f} (For best model)'.format(score[0], score[1]))
    print('Best model stored in {}'.format(cache_model_path))
    return cache_model_path


def evaluate_model(model_path, input_size, train_csv, valid_csv):
    from keras.models import load_model
    print('Load model: {}'.format(model_path))
    model = load_model(model_path)

    batch_size = 1024
    X_train, Y_train, X_valid, Y_valid = load_train_valid_data(train_csv, valid_csv)
    print('Train shape: {}'.format(X_train.shape))
    print('Valid shape: {}'.format(X_valid.shape))
    score = model.evaluate_generator(
        generator=evaluate_generator(X_valid, Y_valid, batch_size, input_size, preprocess_input),
        steps=X_valid.shape[0] // batch_size,
        max_queue_size=10, verbose=1)
    print('Full validation loss: {:.4f} Full validation accuracy: {:.4f} (For best model)'.format(score[0], score[1]))


if __name__ == '__main__':
    type = 'people'
    # type = 'cars'
    # type = 'animals'

    train_csv = CACHE_PATH + 'oid_train_{}.csv'.format(type)
    valid_csv = CACHE_PATH + 'oid_validation_{}.csv'.format(type)
    # best_model_path = train_mobile_net_v1(MOBILENET_INPUT_SIZE, train_csv, valid_csv, type)
    best_model_path = MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_people_loss_0.3600_acc_0.8442_epoch_38.h5'
    evaluate_model(best_model_path, MOBILENET_INPUT_SIZE, train_csv, valid_csv)


'''
Animals MobileNet v1 (0.25, 128px): 
Ep 1: 1563s 16s/step - loss: 0.3970 - acc: 0.8251 - val_loss: 0.4029 - val_acc: 0.8228
Ep 2: 1413s 14s/step - loss: 0.3499 - acc: 0.8453 - val_loss: 0.4436 - val_acc: 0.7914
Ep 3: 1435s 14s/step - loss: 0.3365 - acc: 0.8518 - val_loss: 0.3098 - val_acc: 0.8671
Ep 4: 1425s 14s/step - loss: 0.3279 - acc: 0.8563 - val_loss: 0.3208 - val_acc: 0.8587
Ep 5: 1441s 14s/step - loss: 0.3204 - acc: 0.8597 - val_loss: 0.3832 - val_acc: 0.8334
Ep 6: 1501s 15s/step - loss: 0.3198 - acc: 0.8610 - val_loss: 0.5252 - val_acc: 0.7404
Ep 7: 1506s 15s/step - loss: 0.3170 - acc: 0.8621 - val_loss: 0.3017 - val_acc: 0.8725
Ep 8: 1502s 15s/step - loss: 0.3124 - acc: 0.8643 - val_loss: 0.2960 - val_acc: 0.8740
Ep 14:1464s 15s/step - loss: 0.3020 - acc: 0.8687 - val_loss: 0.2765 - val_acc: 0.8811
Ep 17:1504s 15s/step - loss: 0.2951 - acc: 0.8712 - val_loss: 0.2786 - val_acc: 0.8853
Ep 18:1495s 15s/step - loss: 0.2944 - acc: 0.8733 - val_loss: 0.3067 - val_acc: 0.8671
Ep 19:1547s 15s/step - loss: 0.2923 - acc: 0.8738 - val_loss: 0.3065 - val_acc: 0.8663
Ep 33:1687s 17s/step - loss: 0.2748 - acc: 0.8826 - val_loss: 0.2486 - val_acc: 0.8967 - best
Ep 35:1750s 18s/step - loss: 0.2751 - acc: 0.8817 - val_loss: 0.2686 - val_acc: 0.8831
Ep 38:1789s 18s/step - loss: 0.2729 - acc: 0.8822 - val_loss: 0.2726 - val_acc: 0.8883
Ep 39:1814s 18s/step - loss: 0.2722 - acc: 0.8837 - val_loss: 0.2951 - val_acc: 0.8752
Full validation loss: 0.2788 Full validation accuracy: 0.8866 (For best model)

Cars MobileNet v1 (0.25, 128px): 
Ep 1: 1566s 16s/step - loss: 0.3212 - acc: 0.8624 - val_loss: 0.1829 - val_acc: 0.9324
Ep 2: 1429s 14s/step - loss: 0.2557 - acc: 0.8907 - val_loss: 0.1463 - val_acc: 0.9472
Ep 3: 1564s 16s/step - loss: 0.2438 - acc: 0.8968 - val_loss: 0.1586 - val_acc: 0.9424
Ep 4: 1566s 16s/step - loss: 0.2360 - acc: 0.9003 - val_loss: 0.1453 - val_acc: 0.9473
Ep 17:1664s 17s/step - loss: 0.1947 - acc: 0.9208 - val_loss: 0.1188 - val_acc: 0.9586 LR: 0.00090
Ep 21:1726s 17s/step - loss: 0.1908 - acc: 0.9214 - val_loss: 0.1240 - val_acc: 0.9552
Ep 24:1739s 17s/step - loss: 0.1819 - acc: 0.9261 - val_loss: 0.1183 - val_acc: 0.9567 LR: 0.00081
Ep 25:1747s 17s/step - loss: 0.1841 - acc: 0.9246 - val_loss: 0.1406 - val_acc: 0.9500
Ep 47:2232s 22s/step - loss: 0.1657 - acc: 0.9336 - val_loss: 0.1188 - val_acc: 0.9597 
Ep 48:1713s 17s/step - loss: 0.1720 - acc: 0.9299 - val_loss: 0.1183 - val_acc: 0.9556
Ep 49:1544s 15s/step - loss: 0.1662 - acc: 0.9321 - val_loss: 0.1265 - val_acc: 0.9564 
Ep 55:1558s 16s/step - loss: 0.1659 - acc: 0.9334 - val_loss: 0.1134 - val_acc: 0.9613 LR: 0.00045
Ep 67:1584s 16s/step - loss: 0.1576 - acc: 0.9355 - val_loss: 0.1088 - val_acc: 0.9631 LR: 0.0003645 - Best
Ep 72:1625s 16s/step - loss: 0.1528 - acc: 0.9393 - val_loss: 0.1273 - val_acc: 0.9594 LR: 0.00032805
Full validation loss: 0.0993 Full validation accuracy: 0.9662 (For best model)

People MobileNet v1 (0.25, 128px):
Ep 1: 1716s 17s/step - loss: 0.4718 - acc: 0.7887 - val_loss: 0.7406 - val_acc: 0.7069 
Ep 2: 1550s 15s/step - loss: 0.3771 - acc: 0.8368 - val_loss: 0.5902 - val_acc: 0.7573
Ep 20:1625s 16s/step - loss: 0.3022 - acc: 0.8749 - val_loss: 0.3756 - val_acc: 0.8370
Ep 25:1710s 17s/step - loss: 0.2953 - acc: 0.8794 - val_loss: 0.3769 - val_acc: 0.8374
Ep 30:1726s 17s/step - loss: 0.2953 - acc: 0.8775 - val_loss: 0.3741 - val_acc: 0.8411
Ep 32:1748s 17s/step - loss: 0.2911 - acc: 0.8813 - val_loss: 0.3645 - val_acc: 0.8442
Ep 38:1818s 18s/step - loss: 0.2844 - acc: 0.8844 - val_loss: 0.3600 - val_acc: 0.8442 - best
Ep 41:1871s 19s/step - loss: 0.2891 - acc: 0.8810 - val_loss: 0.3963 - val_acc: 0.8322 LR: 8.100000122794882e-05
Ep 43:1895s 19s/step - loss: 0.2851 - acc: 0.8838 - val_loss: 0.3851 - val_acc: 0.8361 LR: 7.289999848580919e-05
Ep 52:1999s 20s/step - loss: 0.2805 - acc: 0.8852 - val_loss: 0.3790 - val_acc: 0.8433
Ep 68:2276s 23s/step - loss: 0.2795 - acc: 0.8862 - val_loss: 0.4114 - val_acc: 0.8298 LR: 4.304672074795235e-05
Full validation loss: 0.3053 Full validation accuracy: 0.8739 (For best model)
'''