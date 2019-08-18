# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

'''
Code to find reduction coefficients for fixed point representation of weights.
It run some images from validation part of dataset to find maximum ranges of values.
Then convert RELU6 -> RELU1 and rescale some weights and biases.
At the end code checks that initial and rescaled models gives totally same result.  
'''

import os
import glob
from a01_oid_utils import read_single_image, DATASET_PATH


if __name__ == '__main__':
    # Block to choose backend
    gpu_use = 4
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    print('GPU use: {}'.format(gpu_use))


from keras import backend as K
from a00_common_functions import *


# Coefficient to make safe gap for found range to prevent overflow. Lower - less safe, higher - more rounding error.
GAP_COEFF = 1.0


def preproc_input_mathmodel(x):
    x /= 127.5
    x -= 1.
    return x


def rescale_weights(model, layer_num, coeff):
    w = model.layers[layer_num].get_weights()
    model.layers[layer_num].set_weights(w / coeff)
    return model


def rescale_weights_with_bias(model, layer_num, coeff, current_scale):
    w, b = model.layers[layer_num].get_weights()
    w_new = w / coeff
    b_new = b / (coeff * current_scale)
    model.layers[layer_num].set_weights((w_new, b_new))
    return model


def rescale_only_bias(model, layer_num, coeff, current_scale):
    w, b = model.layers[layer_num].get_weights()
    w_new = w.copy()
    b_new = b / (coeff * current_scale)
    model.layers[layer_num].set_weights((w_new, b_new))
    return model


def rescale_batch_norm_weights_initital_v1(model, layer_num, coeff, current_scale):
    eps = 0.001
    gamma, beta, run_mean, run_std = model.layers[layer_num].get_weights()
    gamma /= (coeff * current_scale)
    beta /= (coeff * current_scale)
    run_mean /= (coeff * current_scale)
    # Из за квадратного корня и EPS тут не всё так просто. Надо пересчитывать по формуле
    # после решения уравнения sqrt(Mx+e) = sqrt(x+e)/K
    # M = 1/(K*K) + e*(1-K*K)/(K*K*x)
    c2 = (coeff*coeff*current_scale*current_scale)
    # print('Run std: {}'.format(run_std))
    run_std = run_std/c2 + eps*(1-c2)/c2
    model.layers[layer_num].set_weights((gamma, beta, run_mean, run_std))
    return model


def rescale_batch_norm_weights_initital(model, layer_num, coeff, current_scale):
    gamma, beta, run_mean, run_std = model.layers[layer_num].get_weights()
    beta /= (coeff * current_scale)
    run_mean /= current_scale
    gamma /= coeff
    model.layers[layer_num].set_weights((gamma, beta, run_mean, run_std))
    return model


def rescale_dense_weights(model, layer_num, current_scale, coeff):
    weights = model.layers[layer_num].get_weights()
    if len(weights) == 2:
        w, b = weights
        w /= coeff
        b /= (current_scale*coeff)
        model.layers[layer_num].set_weights((w, b))
    else:
        w = weights
        w /= coeff
        model.layers[layer_num].set_weights(w)
    return model


def is_next_relu6(model, layer_id):
    if layer_id >= len(model.layers) - 1:
        return False
    layer = model.layers[layer_id + 1]
    layer_type = layer.__class__.__name__
    if layer_type == 'Activation':
        config = layer.get_config()
        activation = config['activation']
        if activation == 'relu6':
            return True
    return False


def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(inputs=layers[0].input, outputs=x)
    return new_model


def get_min_max_for_model(model, img_list):
    from keras.models import Model
    from keras.layers import ReLU
    from keras.models import load_model
    from keras.optimizers import Adam

    reduction_koeffs = dict()
    current_six_value = 1.0
    current_scale = 6.0
    eps = 0.001
    first_rescale = True

    for i in range(len(model.layers)):
        class_name = model.layers[i].__class__.__name__
        layer = model.layers[i]
        print('Layer {}: {} Name {}'.format(i, class_name, layer.name))
        print('In nodes: {}'.format(len(layer._inbound_nodes)))
        w1 = layer.get_weights()
        red_coeff = 1.0
        if len(w1) > 0:
            submodel = Model(inputs=model.inputs, outputs=layer.output)
            print(submodel.summary())
            # out = submodel.predict(img_list)
            if class_name == 'Conv2D':
                config = layer.get_config()
                use_bias = config['use_bias']

                print('Min weights value: {} Max weights value: {}'.format(w1[0].min(), w1[0].max()))
                print('Min bias value: {} Max bias value: {}'.format(w1[1].min(), w1[1].max()))

                if first_rescale is True:
                    model = rescale_weights_with_bias(model, i, 6.0, 1.0)
                    first_rescale = False
                else:
                    model = rescale_only_bias(model, i, red_coeff, current_scale)

            elif class_name == 'DepthwiseConv2D':
                config = layer.get_config()
                print(config)
                use_bias = config['use_bias']

                print('Min weights value: {} Max weights value: {}'.format(w1[0].min(), w1[0].max()))
                print('Min bias value: {} Max bias value: {}'.format(w1[1].min(), w1[1].max()))

                model = rescale_only_bias(model, i, red_coeff, current_scale)

            elif class_name == 'Dense':
                config = layer.get_config()
                use_bias = config['use_bias']
                print('Bias state: {}'.format(use_bias))

                if use_bias == False:
                    print('We dont need to rescale Dense')
                else:
                    print('Bias not supported yet!')
                    exit()
            else:
                continue

            reduction_koeffs[i] = red_coeff
            print('Layer: {} Scale: {} Reduction coeff: {} Six value: {}'.format(i, current_scale, red_coeff, current_six_value))

        if class_name == 'Activation' or class_name == 'ReLU':
            print(layer.get_config())

            # Replace model with new activation
            # model = replace_intermediate_layer_in_keras(model, i, Activation(lambda x: relu(x, max_value=current_six_value), name='custom_relu_{}'.format(i)))
            print('Activation six value: {}'.format(current_six_value))
            if abs(current_six_value - 1.0) > 0.0000001:
                print('Not expected six value!')
                exit()

            # We always add relu_1 activation (due to scaling algorithm)
            model = replace_intermediate_layer_in_keras(model, i, ReLU(max_value=1.0, name='custom_relu_{}'.format(i)))
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            model.save(MODEL_PATH + 'debug.h5')
            model = load_model(MODEL_PATH + 'debug.h5')
            print(model.summary)

        if i == 0:
            continue

        # Check new min, max
        layer = model.layers[i]
        o = layer.output
        submodel = Model(inputs=model.inputs, outputs=o)
        print(submodel.summary())
        out = submodel.predict(img_list)
        print('Rescaled submodel: {} Min out value: {} Max out value: {}'.format(out.shape, out.min(), out.max()))

    print('Reduction koeffs: ', reduction_koeffs)
    return model, reduction_koeffs


def load_oid_data(type):
    from keras.utils import to_categorical
    valid = pd.read_csv(CACHE_PATH + 'oid_validation_{}.csv'.format(type))
    train = pd.read_csv(CACHE_PATH + 'oid_train_{}.csv'.format(type))
    X_train = train['id'].values
    Y_train = to_categorical(train['target'].values, num_classes=2)
    X_valid = valid['id'].values
    Y_valid = to_categorical(valid['target'].values, num_classes=2)
    return  X_train, Y_train, X_valid, Y_valid


def process_single_item(id, box_size):
    img = read_single_image(DATASET_PATH + 'validation/' + id + '.jpg')
    img = cv2.resize(img, (box_size, box_size), interpolation=cv2.INTER_LINEAR)
    return img


def check_results_are_the_same(model_path1, model_path2, img_list):
    from keras.models import load_model
    modelA = load_model(model_path1)
    modelB = load_model(model_path2)

    resA = modelA.predict(img_list)
    resB = modelB.predict(img_list)
    print(resA)
    print(resB)
    print('Probabilities shape: {}'.format(resA.shape))

    maxA = resA.argmax(axis=1)
    maxB = resB.argmax(axis=1)
    print(maxA)
    print(maxB)
    print('Answer shape: {}'.format(maxA.shape))

    print(np.unique(maxA, return_counts=True))
    print(np.unique(maxB, return_counts=True))

    diff = len(maxA[maxA != maxB])
    print('Answer difference: {}'.format(diff))
    print((maxA - maxB).sum())


if __name__ == '__main__':
    from kito import reduce_keras_model
    from keras.models import load_model
    from keras.applications.mobilenet import preprocess_input

    # Params
    image_limit = 10000
    input_size = 128
    model_type = 'animals'
    model_path = MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33.h5'
    model_path_reduced = model_path[:-3] + '_reduced.h5'
    model_path_rescaled = model_path[:-3] + '_reduced_rescaled.h5'

    if not os.path.isfile(model_path_reduced):
        model = load_model(model_path)
        model = reduce_keras_model(model, verbose=True)
        model.save(model_path_reduced)
    else:
        model = load_model(model_path_reduced)
    print(model.summary())
    print('Number of layers: {}'.format(len(model.layers)))

    X_train, Y_train, X_test, Y_test = load_oid_data(model_type)
    print(X_train.shape, X_test.shape)
    X_test = X_test[:image_limit]
    Y_test = Y_test[:image_limit]
    uni = np.unique(Y_test, return_counts=True)
    print(uni[0].sum())

    img_list = []
    for i in range(len(X_test)):
        img = process_single_item(X_test[i], input_size)
        img_list.append(img)
    img_list = np.array(img_list, dtype=np.float32)
    img_list = preprocess_input(img_list)
    print("Image limit: {} Images shape: {}".format(image_limit, img_list.shape))

    model, reduction_koeffs = get_min_max_for_model(model, img_list)

    overall_reduction_rate = 1.0
    for i in sorted(reduction_koeffs.keys()):
        print('Layer {} reduction coeff: {}'.format(i, reduction_koeffs[i]))
        overall_reduction_rate *= reduction_koeffs[i]
    print('Overall scale change: {}'.format(overall_reduction_rate))

    print('Save model in {}'.format(model_path_rescaled))
    model.save(model_path_rescaled)

    check_results_are_the_same(model_path, model_path_rescaled, img_list)
