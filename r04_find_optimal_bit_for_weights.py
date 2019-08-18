# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'


'''
This code finds out which bit size for weight lead to zero classification error on fixed point test data
comparing with floating point test data. Start search from 8 bits up to 32 bits.
'''

if __name__ == '__main__':
    import os

    # Block to choose backend
    gpu_use = 2
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
from scipy.signal import convolve2d
import math
import tensorflow as tf

sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.logging.set_verbosity(tf.logging.ERROR)


# Note: We suppose that every Conv2D layer has type "same"
# In Tensorflow weight matrices already transposed
def my_convolve(input, kernel):
    output = np.zeros((input.shape[0], input.shape[1]))
    zero_pad = np.zeros((input.shape[0] + 2, input.shape[1] + 2))
    zero_pad[1:-1, 1:-1] = input
    # kernel = np.flipud(kernel)
    # kernel = np.fliplr(kernel)
    for i in range(1, zero_pad.shape[0] - 1):
        for j in range(1, zero_pad.shape[1] - 1):
            sub = zero_pad[i-1:i+2, j-1:j+2]
            output[i-1, j-1] = np.sum(sub*kernel)
    return output


def my_convolve_fixed_point(input, kernel, bit):
    output = np.zeros((input.shape[0], input.shape[1]))
    zero_pad = np.zeros((input.shape[0] + 2, input.shape[1] + 2))
    zero_pad[1:-1, 1:-1] = input
    # kernel = np.flipud(kernel)
    # kernel = np.fliplr(kernel)
    for i in range(1, zero_pad.shape[0] - 1):
        for j in range(1, zero_pad.shape[1] - 1):
            sub = zero_pad[i-1:i+2, j-1:j+2]
            output[i-1, j-1] = np.sum((sub*kernel).astype(np.int64))
    return output


def preprocess_forward(arr, val):
    arr1 = arr.copy().astype(np.float32)
    arr1 /= val
    return arr1


def convert_to_fix_point(arr1, bit):
    arr2 = arr1.copy().astype(np.float32)
    arr2[arr2 < 0] = 0.0
    arr2 = np.round(np.abs(arr2) * (2 ** bit))
    arr3 = arr1.copy().astype(np.float32)
    arr3[arr3 > 0] = 0.0
    arr3 = -np.round(np.abs(-arr3) * (2 ** bit))
    arr4 = arr2 + arr3
    return arr4.astype(np.int64)


def from_fix_point_to_float(arr, bit):
    return arr / (2 ** bit)


def compare_outputs(s1, s2, debug_info=True):
    if s1.shape != s2.shape:
        print('Shape of arrays is different! {} != {}'.format(s1.shape, s2.shape))
    s = np.abs(s1 - s2)
    size = 1
    for dim in np.shape(s): size *= dim
    if debug_info:
        print('Max difference: {}'.format(s.max()))
        print('Avg difference: {}'.format(s.mean()))
        print('Value range float: {} - {}'.format(s1.min(), s1.max()))
        print('Value range fixed: {} - {}'.format(s2.min(), s2.max()))


def print_first_pixel_detailed_calculation_dense(previous_layer_output, wgt_bit, bit_precizion):
    i = 10
    conv_my = 0
    for j in range(0, previous_layer_output.shape[0]):
        print('Pixel {}: {}'.format(j, int(previous_layer_output[j])))
        print('Weight {}: {}'.format(j, wgt_bit[j][i]))
        conv_my += np.right_shift((previous_layer_output[j]*wgt_bit[j][i]).astype(np.int64), bit_precizion)
        if j > 0 and j % 9 == 8:
            print('Current conv_my: {}'.format(conv_my))
    print('Result first pixel: {}'.format(conv_my))
    exit()


def print_first_pixel_detailed_calculation(previous_layer_output, wgt_bit, bit_precizion):
    i = 0
    x = 0
    y = 0
    conv_my = 0
    print('Point: {} X: {} Y: {}'.format(i, x, y))
    print('Weights shape: {}'.format(wgt_bit.shape))
    for j in range(wgt_bit.shape[2]):
        full_image = previous_layer_output[:, :, j]
        zero_pad = np.zeros((full_image.shape[0] + 2, full_image.shape[1] + 2))
        zero_pad[1:-1, 1:-1] = full_image
        pics = zero_pad[x+1-1:x+1+2, y+1-1:y+1+2].astype(np.int64)
        print('Pixel area 3x3 for [{}, {}]:'.format(x, y), pics)
        kernel = wgt_bit[:, :, j, i].copy()
        # Не надо переворачивать для TensorFlow
        # kernel = np.flipud(kernel)
        # kernel = np.fliplr(kernel)
        print('Weights {}: {}'.format(j, kernel))
        res = np.sum(np.right_shift((pics*kernel).astype(np.int64), bit_precizion))
        print('Convolution result {}: {}'.format(j, res))
        conv_my += res

    print('Overall result: {}'.format(conv_my))
    if conv_my[conv_my > 2 ** bit_precizion].any() or conv_my[conv_my < - 2 ** bit_precizion].any():
        print('Overflow! {}'.format(conv_my[conv_my > 2 ** bit_precizion]))
        exit()
    if conv_my < 0:
        conv_my = 0
    exit()


def mmZeroPadding2D_floating_point(layer, img):
    config = layer.get_config()
    print(config)
    if len(config['padding']) == 1:
        padding1_start = config['padding']
        padding1_end = config['padding']
        padding2_start = config['padding']
        padding2_end = config['padding']
    elif len(config['padding']) == 2:
        padding1_start = config['padding'][0][0]
        padding1_end = config['padding'][0][1]
        padding2_start = config['padding'][1][0]
        padding2_end = config['padding'][1][1]
    out = np.zeros((img.shape[0],
                    img.shape[1] + padding1_start + padding1_end,
                    img.shape[2] + padding2_start + padding2_end,
                    img.shape[3]), dtype=np.float64)
    out[:, padding1_start:out.shape[1] - padding1_end, padding2_start:out.shape[2] - padding2_end, :] = img.copy()
    return out


def mmZeroPadding2D_fixed_point(layer, img):
    config = layer.get_config()
    print(config)
    if len(config['padding']) == 1:
        padding1_start = config['padding']
        padding1_end = config['padding']
        padding2_start = config['padding']
        padding2_end = config['padding']
    elif len(config['padding']) == 2:
        padding1_start = config['padding'][0][0]
        padding1_end = config['padding'][0][1]
        padding2_start = config['padding'][1][0]
        padding2_end = config['padding'][1][1]
    out = np.zeros((img.shape[0],
                    img.shape[1] + padding1_start + padding1_end,
                    img.shape[2] + padding2_start + padding2_end,
                    img.shape[3]), dtype=np.int64)
    out[:, padding1_start:out.shape[1] - padding1_end, padding2_start:out.shape[2] - padding2_end, :] = img.copy()
    return out


def run_TF_Conv2D(img, w, b, strides, padding, type='float'):
    global sess
    in1 = tf.Variable(img.astype(np.float64))
    w1 = tf.Variable(w.astype(np.float64))
    b1 = tf.Variable(b.astype(np.float64))
    data = tf.nn.conv2d(in1, w1, (1,) + strides + (1,), str(padding).upper())
    data = tf.nn.bias_add(data, b1)
    sess.run(tf.global_variables_initializer())
    out = sess.run(data)
    if type == 'float':
        out = out.astype(np.float64)
    else:
        out = out.astype(np.int64)
    tf.reset_default_graph()
    sess = tf.Session()
    return out


def run_TF_Depthwise_Conv2D(img, w, b, strides, padding, type='float'):
    global sess
    in1 = tf.Variable(img.astype(np.float64))
    w1 = tf.Variable(w.astype(np.float64))
    b1 = tf.Variable(b.astype(np.float64))
    data = tf.nn.depthwise_conv2d(in1, w1, (1,) + strides + (1,), str(padding).upper())
    data = tf.nn.bias_add(data, b1)
    sess.run(tf.global_variables_initializer())
    out = sess.run(data)
    if type == 'float':
        out = out.astype(np.float64)
    else:
        out = out.astype(np.int64)
    tf.reset_default_graph()
    sess = tf.Session()
    return out


def mmConv2D_floating_point(layer, img, debug_info):
    global sess
    calc_type = 'tf'
    config = layer.get_config()
    filters = config['filters']
    use_bias = config['use_bias']
    strides = config['strides']
    padding = config['padding']
    kernel_size = config['kernel_size']
    if debug_info and 0:
        print(config)

    sh1 = img.shape[1]
    sh2 = img.shape[2]
    if padding == 'valid':
        sh1 -= 2 - (img.shape[1] % 2)
        sh2 -= 2 - (img.shape[2] % 2)

    if strides == (1, 1):
        out = np.zeros((img.shape[0], sh1, sh2, filters), dtype=np.float64)
    elif strides == (2, 2):
        out = np.zeros((img.shape[0], sh1 // 2, sh2 // 2, filters), dtype=np.float64)
        # calc_type = 'slow'
    else:
        print('Not supported conditions yet!')
        exit()

    if kernel_size != (3, 3) and kernel_size != (1, 1):
        print('Unsupported kernel size: {}'.format(kernel_size))
        exit()

    (w, b) = layer.get_weights()

    if debug_info:
        print(w.shape, b.shape, out.shape)

    if calc_type == 'slow':
        # Cycle by different batch images
        for sh0 in range(img.shape[0]):
            # output filters cycle
            for wi in range(w.shape[-1]):
                # input filters cycle
                for wj in range(w.shape[-2]):
                    kernel = w[:, :, wj, wi].copy()
                    slice = img[sh0, :, :, wj]

                    if padding == 'same':
                        zero_pad = np.zeros((slice.shape[0] + 2, slice.shape[1] + 2))
                        zero_pad[1:-1, 1:-1] = slice
                    elif padding == 'valid':
                        zero_pad = slice.copy()
                    else:
                        print('Unknown padding: {}'.format(padding))
                        exit()
                    # convolution
                    for i in range(1, zero_pad.shape[0] - 1, strides[0]):
                        for j in range(1, zero_pad.shape[1] - 1, strides[0]):
                            if kernel_size == (3, 3):
                                sub = zero_pad[i - 1:i + 2, j - 1:j + 2]
                                out[sh0, (i - 1) // strides[0], (j - 1) // strides[1], wi] += np.sum(sub * kernel)
                            elif kernel_size == (1, 1):
                                sub = zero_pad[i, j]
                                out[sh0, (i - 1) // strides[0], (j - 1) // strides[1], wi] += sub * kernel[0, 0]
                out[sh0, :, :, wi] += b[wi]
    elif calc_type == 'fast':
        # Cycle by different batch images
        for sh0 in range(img.shape[0]):
            # output filters cycle
            for wi in range(w.shape[-1]):
                # input filters cycle
                for wj in range(w.shape[-2]):
                    kernel = w[:, :, wj, wi].copy()
                    slice = img[sh0, :, :, wj].copy()
                    conv_my = convolve2d(slice, kernel, mode=padding)
                    out[sh0, :, :, wi] += conv_my
                out[sh0, :, :, wi] += b[wi]
    elif calc_type == 'tf':
        out[...] = run_TF_Conv2D(img, w, b, strides, padding, 'float')

    return out


def mmConv2D_fixed_point(layer, img, bit_precizion, bit_precizion_weights, bit_precizion_bias, debug_info):
    global sess
    calc_type = 'tf'
    # Convolution with fixed point
    config = layer.get_config()
    filters = config['filters']
    use_bias = config['use_bias']
    strides = config['strides']
    padding = config['padding']
    kernel_size = config['kernel_size']
    if debug_info and 0:
        print(config)

    sh1 = img.shape[1]
    sh2 = img.shape[2]
    if padding == 'valid':
        sh1 -= 2 - (img.shape[1] % 2)
        sh2 -= 2 - (img.shape[2] % 2)

    if strides == (1, 1):
        out = np.zeros((img.shape[0], sh1, sh2, filters), dtype=np.int64)
    elif strides == (2, 2):
        out = np.zeros((img.shape[0], sh1 // 2, sh2 // 2, filters), dtype=np.int64)
    else:
        print('Not supported conditions yet!')
        exit()

    if kernel_size != (3, 3) and kernel_size != (1, 1):
        print('Unsupported kernel size: {}'.format(kernel_size))
        exit()

    (w, b) = layer.get_weights()

    if debug_info:
        print(w.shape, b.shape, out.shape)

    w = convert_to_fix_point(w.copy(), bit_precizion_weights)
    b = convert_to_fix_point(b.copy(), bit_precizion_bias)

    # We need to shift it to sum with result of multiplication
    b <<= bit_precizion_weights + (bit_precizion - bit_precizion_bias)


    if calc_type == 'slow':
        # Cycle by different batch images
        for sh0 in range(img.shape[0]):
            # output filters cycle
            for wi in range(w.shape[-1]):
                # input filters cycle
                for wj in range(w.shape[-2]):
                    kernel = w[:, :, wj, wi].copy()
                    slice = img[sh0, :, :, wj]

                    if padding == 'same':
                        zero_pad = np.zeros((slice.shape[0] + 2, slice.shape[1] + 2))
                        zero_pad[1:-1, 1:-1] = slice
                    elif padding == 'valid':
                        zero_pad = slice.copy()
                    else:
                        print('Unknown padding: {}'.format(padding))
                        exit()

                    # convolution
                    for i in range(1, zero_pad.shape[0] - 1, strides[0]):
                        for j in range(1, zero_pad.shape[1] - 1, strides[0]):
                            if kernel_size == (3, 3):
                                sub = zero_pad[i - 1:i + 2, j - 1:j + 2]
                                out[sh0, (i - 1) // strides[0], (j - 1) // strides[1], wi] += np.sum((sub*kernel).astype(np.int64))
                            elif kernel_size == (1, 1):
                                sub = zero_pad[i, j]
                                out[sh0, (i - 1) // strides[0], (j - 1) // strides[1], wi] += (sub*kernel[0, 0]).astype(np.int64)

                out[sh0, :, :, wi] += b[wi]

    elif calc_type == 'fast':
        # Cycle by different batch images
        for sh0 in range(img.shape[0]):
            # output filters cycle
            for wi in range(w.shape[-1]):
                # input filters cycle
                for wj in range(w.shape[-2]):
                    kernel = w[:, :, wj, wi].copy()
                    slice = img[sh0, :, :, wj].copy()
                    conv_my = convolve2d(slice, kernel, mode=padding)
                    out[sh0, :, :, wi] += conv_my
                out[sh0, :, :, wi] += b[wi]
    elif calc_type == 'tf':
        out[...] = run_TF_Conv2D(img, w, b, strides, padding, 'int')

    # Shift it back to initial scale
    out = np.right_shift(out.astype(np.int64), bit_precizion_weights)

    return out


def mmGlobalAveragePooling2D_floating_point(img):
    # Standard glob pool
    result = np.zeros((img.shape[0], img.shape[-1]))
    for j in range(img.shape[0]):
        for i in range(img.shape[-1]):
            result[j, i] = img[j, :, :, i].mean()
    return result


def mmGlobalAveragePooling2D_fixed_point(img):
    # Standard glob pool
    result = np.zeros((img.shape[0], img.shape[-1]), dtype=np.int64)
    block_size = img.shape[1] * img.shape[2]
    for j in range(img.shape[0]):
        for i in range(img.shape[-1]):
            value = img[j, :, :, i].sum() // block_size
            result[j, i] = value
    return result


def mmActivation_floating_point(layer, img, one_value=1.0, debug_info=False):
    config = layer.get_config()
    activation = config['activation']

    if activation != 'relu_1':
        print('Unsupported activation {}!'.format(activation))
        exit()

    result = img.copy()
    result[result < 0] = 0.
    result[result > one_value] = one_value
    return result


def mmActivation_fixed_point(layer, img, bit_precizion, debug_info=False):
    config = layer.get_config()
    activation = config['activation']

    if activation != 'relu_1':
        print('Unsupported activation {}!'.format(activation))
        exit()

    result = img.copy()
    result[result < 0] = 0.
    result[result >= 2 ** bit_precizion] = 2 ** bit_precizion - 1
    return result


def mmReLU_floating_point(layer, img, one_value=1.0, debug_info=False):
    config = layer.get_config()
    max_value = config['max_value']

    if max_value != 1:
        print('Unsupported value for ReLU activation {}!'.format(max_value))
        exit()

    result = img.copy()
    result[result < 0] = 0.
    result[result > one_value] = one_value
    return result


def mmReLU_fixed_point(layer, img, bit_precizion, debug_info=False):
    config = layer.get_config()
    max_value = config['max_value']

    if max_value != 1:
        print('Unsupported value for ReLU activation {}!'.format(max_value))
        exit()

    result = img.copy()
    result[result < 0] = 0.
    result[result >= 2 ** bit_precizion] = 2 ** bit_precizion - 1
    return result


def mmDepthwiseConv2D_floating_point(layer, img, debug_info):
    config = layer.get_config()
    calc_type = 'tf'
    # print(config)
    use_bias = config['use_bias']
    strides = config['strides']
    padding = config['padding']
    kernel_size = config['kernel_size']
    filters = img.shape[3]

    sh1 = img.shape[1]
    sh2 = img.shape[2]
    if padding == 'valid':
        sh1 -= 2 - (img.shape[1] % 2)
        sh2 -= 2 - (img.shape[2] % 2)

    if strides == (1, 1):
        out = np.zeros((img.shape[0], sh1, sh2, filters), dtype=np.float64)
    elif strides == (2, 2):
        out = np.zeros((img.shape[0], sh1 // 2, sh2 // 2, filters), dtype=np.float64)
    else:
        print('Not supported strides yet: {}'.format(strides))
        exit()

    if kernel_size != (3, 3):
        print('Unsupported kernel size: {}'.format(kernel_size))
        exit()

    (w, b) = layer.get_weights()

    print(w.shape, b.shape, out.shape)

    if calc_type == 'slow':
        # Cycle by different batch images
        for sh0 in range(img.shape[0]):
            # input filters cycle
            for wj in range(w.shape[-2]):
                kernel = w[:, :, wj, 0].copy()
                slice = img[sh0, :, :, wj]

                if padding == 'same':
                    zero_pad = np.zeros((slice.shape[0] + 2, slice.shape[1] + 2))
                    zero_pad[1:-1, 1:-1] = slice
                elif padding == 'valid':
                    zero_pad = slice.copy()
                else:
                    print('Unknown padding: {}'.format(padding))
                    exit()
                # kernel = np.flipud(kernel)
                # kernel = np.fliplr(kernel)
                # convolution
                for i in range(1, zero_pad.shape[0] - 1, strides[0]):
                    for j in range(1, zero_pad.shape[1] - 1, strides[0]):
                        sub = zero_pad[i - 1:i + 2, j - 1:j + 2]
                        # print((i - 1) // strides[0], (j - 1) // strides[1], wi)
                        out[sh0, (i - 1) // strides[0], (j - 1) // strides[1], wj] = np.sum(sub * kernel)
                out[sh0, :, :, wj] += b[wj]
    elif calc_type == 'tf':
        out[...] = run_TF_Depthwise_Conv2D(img, w, b, strides, padding, 'float')

    return out


def mmDepthwiseConv2D_fixed_point(layer, img, bit_precizion, bit_precizion_weights, bit_precizion_bias, debug_info):
    config = layer.get_config()
    calc_type = 'tf'
    # print(config)
    use_bias = config['use_bias']
    strides = config['strides']
    padding = config['padding']
    kernel_size = config['kernel_size']
    filters = img.shape[3]

    sh1 = img.shape[1]
    sh2 = img.shape[2]
    if padding == 'valid':
        sh1 -= 2 - (img.shape[1] % 2)
        sh2 -= 2 - (img.shape[2] % 2)

    if strides == (1, 1):
        out = np.zeros((img.shape[0], sh1, sh2, filters), dtype=np.float64)
    elif strides == (2, 2):
        out = np.zeros((img.shape[0], sh1 // 2, sh2 // 2, filters), dtype=np.float64)
    else:
        print('Not supported strides yet: {}'.format(strides))
        exit()

    if kernel_size != (3, 3):
        print('Unsupported kernel size: {}'.format(kernel_size))
        exit()

    (w, b) = layer.get_weights()

    print(w.shape, b.shape, out.shape)

    w = convert_to_fix_point(w.copy(), bit_precizion_weights)
    b = convert_to_fix_point(b.copy(), bit_precizion_bias)

    # We need to shift it to sum with result of multiplication
    b <<= bit_precizion_weights + (bit_precizion - bit_precizion_bias)

    if calc_type == 'slow':
        # Cycle by different batch images
        for sh0 in range(img.shape[0]):
            # input filters cycle
            for wj in range(w.shape[-2]):
                kernel = w[:, :, wj, 0].copy()
                slice = img[sh0, :, :, wj]

                if padding == 'same':
                    zero_pad = np.zeros((slice.shape[0] + 2, slice.shape[1] + 2))
                    zero_pad[1:-1, 1:-1] = slice
                elif padding == 'valid':
                    zero_pad = slice.copy()
                else:
                    print('Unknown padding: {}'.format(padding))
                    exit()
                # kernel = np.flipud(kernel)
                # kernel = np.fliplr(kernel)
                # convolution
                for i in range(1, zero_pad.shape[0] - 1, strides[0]):
                    for j in range(1, zero_pad.shape[1] - 1, strides[0]):
                        sub = zero_pad[i - 1:i + 2, j - 1:j + 2]
                        # print((i - 1) // strides[0], (j - 1) // strides[1], wi)
                        out[sh0, (i - 1) // strides[0], (j - 1) // strides[1], wj] = np.sum((sub*kernel).astype(np.int64))
                out[sh0, :, :, wj] += b[wj]
    elif calc_type == 'tf':
        out[...] = run_TF_Depthwise_Conv2D(img, w, b, strides, padding, 'int')

    # Shift it back to initial scale
    out = np.right_shift(out.astype(np.int64), bit_precizion_weights)

    return out


def mmDense_floating_point(layer, img, debug_info):
    config = layer.get_config()
    print(config)
    use_bias = config['use_bias']
    activation = config['activation']
    units = config['units']
    batch_size = img.shape[0]

    if use_bias:
        (w, b) = layer.get_weights()
    else:
        (w,) = layer.get_weights()

    print('Dense weights shape: {}'.format(w.shape))

    if activation != 'softmax':
        print('Activation {} is not supported'.format(activation))
        exit()

    if use_bias is True:
        print('Bias currently not supported!')
        exit()

    out = np.zeros((batch_size, units))
    for sh0 in range(batch_size):
        for i in range(w.shape[1]):
            for j in range(w.shape[0]):
                out[sh0, i] += img[sh0, j] * w[j, i]

    # Softmax activation part
    # We skip it here because we will use max at the end
    if 0:
        for sh0 in range(batch_size):
            maxy = out[sh0].max()
            out[sh0] = np.exp(out[sh0] - maxy)
            sum = out[sh0].sum()
            out[sh0] /= sum

    return out


def mmDense_fixed_point(layer, img, bit_precizion, bit_precizion_weights, debug_info):
    config = layer.get_config()
    if debug_info is True:
        print(config)
    use_bias = config['use_bias']
    activation = config['activation']
    units = config['units']
    batch_size = img.shape[0]

    if use_bias:
        (w, b) = layer.get_weights()
    else:
        (w,) = layer.get_weights()

    if use_bias is True:
        print('Bias currently not supported!')
        exit()

    if debug_info is True:
        print('Dense weights shape: {}'.format(w.shape))

    if activation != 'softmax':
        print('Activation {} is not supported'.format(activation))
        exit()

    w = convert_to_fix_point(w.copy(), bit_precizion_weights)

    out = np.zeros((batch_size, units))
    for sh0 in range(batch_size):
        for i in range(w.shape[1]):
            for j in range(w.shape[0]):
                out[sh0, i] += img[sh0, j] * w[j, i]

    # Divide by 2^bp
    out = np.right_shift(out.astype(np.int64), bit_precizion_weights)

    if out[out > 2 ** bit_precizion].any() or out[out < - 2 ** bit_precizion].any():
        if out[out > 2 ** bit_precizion].any():
            print('Warning overflow on current level! {}'.format(out[out > 2 ** bit_precizion]))
        else:
            print('Warning overflow on current level! {}'.format(out[out < - 2 ** bit_precizion]))
        print('Max is {}'.format(2 ** bit_precizion))

    # We don't need to find softmax here, since we only need the
    # position of max value, which will be the same

    return out


# bit_precizion - fixed point accuracy in bits
def go_mat_model(model, images, bit_precizion, bit_precizion_weights, bit_precizion_bias, debug_info=True):

    level_out = dict()
    level_out_reduced = dict()
    print_pixel_calc = False

    # Hack before we solve problem with exact 1.0 value
    one_value = (2 ** bit_precizion - 1) / (2 ** bit_precizion)

    for level_id in range(len(model.layers)):
        layer = model.layers[level_id]
        layer_type = layer.__class__.__name__
        if debug_info:
            print('Layer name: {} Layer type: {}'.format(layer.name, layer_type))
            if level_id > 0:
                print('Input shape: {}'.format(level_out[level_id-1].shape))

        if layer_type == 'InputLayer':
            level_out[level_id] = images.copy()
            level_out_reduced[level_id] = convert_to_fix_point(images.copy(), bit_precizion)

        elif layer_type == 'ZeroPadding2D':
            level_out[level_id] = mmZeroPadding2D_floating_point(layer, level_out[level_id - 1].copy())
            level_out_reduced[level_id] = mmZeroPadding2D_fixed_point(layer, level_out_reduced[level_id - 1].copy())

        elif layer_type == 'Conv2D':
            level_out[level_id] = mmConv2D_floating_point(layer, level_out[level_id - 1].copy(), debug_info)
            level_out_reduced[level_id] = mmConv2D_fixed_point(layer, level_out_reduced[level_id - 1].copy(), bit_precizion, bit_precizion_weights, bit_precizion_bias, debug_info)

        elif layer_type == 'DepthwiseConv2D':
            level_out[level_id] = mmDepthwiseConv2D_floating_point(layer, level_out[level_id - 1].copy(), debug_info)
            level_out_reduced[level_id] = mmDepthwiseConv2D_fixed_point(layer, level_out_reduced[level_id - 1].copy(), bit_precizion, bit_precizion_weights, bit_precizion_bias, debug_info)

        elif layer_type == 'Activation':
            level_out[level_id] = mmActivation_floating_point(layer, level_out[level_id - 1].copy(), one_value=one_value, debug_info=debug_info)
            level_out_reduced[level_id] = mmActivation_fixed_point(layer, level_out_reduced[level_id - 1].copy(), bit_precizion, debug_info)

        elif layer_type == 'ReLU':
            level_out[level_id] = mmReLU_floating_point(layer, level_out[level_id - 1].copy(), one_value=one_value, debug_info=debug_info)
            level_out_reduced[level_id] = mmReLU_fixed_point(layer, level_out_reduced[level_id - 1].copy(), bit_precizion, debug_info)

        elif layer_type == 'GlobalAveragePooling2D':
            level_out[level_id] = mmGlobalAveragePooling2D_floating_point(level_out[level_id - 1].copy())
            level_out_reduced[level_id] = mmGlobalAveragePooling2D_fixed_point(level_out_reduced[level_id - 1].copy())

        elif layer_type == 'Dense':
            level_out[level_id] = mmDense_floating_point(layer, level_out[level_id - 1].copy(), debug_info)
            level_out_reduced[level_id] = mmDense_fixed_point(layer, level_out_reduced[level_id - 1].copy(),
                                                                        bit_precizion, bit_precizion_weights, debug_info)

        # Convert back to float for comparison
        checker_tmp = from_fix_point_to_float(level_out_reduced[level_id], bit_precizion)
        compare_outputs(level_out[level_id], checker_tmp, debug_info)

        if debug_info:
            print('')

        if level_id > 1000:
            exit()

        if layer.name == 'conv_dw_2_bn_':
            exit()

    print(level_out[len(model.layers) - 1].shape)
    print(level_out[len(model.layers) - 1])
    print(level_out_reduced[len(model.layers) - 1].shape)
    print(level_out_reduced[len(model.layers) - 1])
    pred_float = np.argmax(level_out[len(model.layers) - 1], axis=1)
    pred_fixed = np.argmax(level_out_reduced[len(model.layers) - 1], axis=1)
    error_rate = (pred_float != pred_fixed).sum() / level_out[len(model.layers) - 1].shape[0]

    return error_rate, pred_float, pred_fixed


def get_error_rate(a1, a2):
    miss = 0
    for i in range(len(a1)):
        if a1[i] != a2[i]:
            miss += 1
    print('Error rate: {}%'.format(round(100*miss/len(a1), 2)))
    return miss


def preproc_input_mathmodel(x):
    x -= 127.5
    x /= 128.
    return x


def load_oid_data_optimal(type):
    valid = pd.read_csv(CACHE_PATH + 'oid_validation_{}.csv'.format(type))
    X_valid = valid['id'].values
    Y_valid = valid['target'].values
    return X_valid, Y_valid


def get_image_set(type, image_limit, preproc_type='keras'):
    from keras.applications.mobilenet import preprocess_input
    from r03_mobilenet_v1_reduce_and_scale_model import process_single_item
    from a01_oid_utils import read_single_image, DATASET_PATH

    input_size = 128
    X_test, Y_test = load_oid_data_optimal(type)
    condition1 = (Y_test == 0)
    print(X_test.shape, Y_test.shape)
    X_test = np.concatenate((
        X_test[condition1][:image_limit // 2],
        X_test[~condition1][:image_limit // 2],
    ))
    Y_test = np.concatenate((
        Y_test[condition1][:image_limit // 2],
        Y_test[~condition1][:image_limit // 2],
    ))

    print(X_test.shape)
    uni = np.unique(Y_test, return_counts=True)
    print('Targets: {}'.format(uni))

    img_list = []
    for i in range(len(X_test)):
        img = process_single_item(X_test[i], input_size)
        img_list.append(img)
    img_list = np.array(img_list, dtype=np.float32)
    if preproc_type == 'keras':
        img_list = preprocess_input(img_list)
    else:
        img_list = preproc_input_mathmodel(img_list)
    print("Image limit: {} Images shape: {}".format(image_limit, img_list.shape))
    return img_list, Y_test


def find_conv_overflow_bit_values(model):
    max_w = -1000000000
    max_b = -1000000000
    for level_id in range(len(model.layers)):
        layer = model.layers[level_id]
        layer_type = layer.__class__.__name__
        if layer_type == 'Conv2D' or layer_type == 'DepthwiseConv2D':
            print('Go for layer: {}'.format(layer.name))
            config = layer.get_config()
            w, b = layer.get_weights()
            print('Weights range: {} - {}'.format(w.min(), w.max()))
            print('Bias range: {} - {}'.format(w.min(), w.max()))
            if w.max() > max_w:
                max_w = w.max()
            if np.abs(w.min()) > max_w:
                max_w = np.abs(w.min())
            if b.max() > max_b:
                max_b = b.max()
            if np.abs(b.min()) > max_b:
                max_b = np.abs(b.min())
    print('Maximum weight in covolution overall: {}'.format(max_w))
    print('Maximum bias in covolution overall: {}'.format(max_b))
    max_w_bit = math.ceil(math.log(max_w, 2))
    max_b_bit = math.ceil(math.log(max_b, 2))
    print('Overflow for conv weights w: {} bits b: {} bits'.format(max_w_bit, max_b_bit))

    return max_w_bit, max_b_bit


# This function works slow, so it should be run once to find optimal bit
def get_optimal_bit_for_weights(type, model_path, image_limit, acceptable_error_rate, use_cache):
    cache_path = CACHE_PATH + 'optimal_bit_{}_{}.pklz'.format(type, image_limit)
    if not os.path.isfile(cache_path) or use_cache is not True:
        print('Read model...')
        # We read already reduced weights. We don't need to fix them any way
        model = get_model(model_path)
        print(model.summary())
        convW, convB = find_conv_overflow_bit_values(model)

        # We doing preprocessing a little bit different because values shouldn't goes to -1 and 1 values (it will lead to overflow).
        # It can reduce accuracy a little bit. Probably we should initially train with this preproc
        images, answers = get_image_set(type, image_limit, 'math')

        print('Classify images...')
        keras_out = model.predict(images)
        res_keras_array = []
        acc = 0.
        for i in range(keras_out.shape[0]):
            res_keras_array.append(np.argmax(keras_out[i]))
            if res_keras_array[-1] == answers[i]:
               acc += 1.
        print('Keras result raw: ', keras_out)
        print('Keras result pos: ', res_keras_array)
        print('Accuracy: {}'.format(acc / keras_out.shape[0]))

        image_bit_precision = 8
        weight_bit_precision = 16
        bias_bit_precision = 16

        if 1:
            print('First run')
            while 1:
                print('\nStart image bit precision: {} Weights precision: {} Bias precision: {}'.format(image_bit_precision, weight_bit_precision, bias_bit_precision))
                error_rate, pred_float, pred_fixed = go_mat_model(model, images, image_bit_precision, weight_bit_precision, bias_bit_precision, debug_info=True)
                print('Error rate: {:.6f}'.format(error_rate))
                print(res_keras_array)
                print(pred_float)
                print(pred_fixed)
                image_bit_precision += 1
                if error_rate < acceptable_error_rate or image_bit_precision > 36:
                    break

            if image_bit_precision > 32:
                return -1, -1, -1, -1, -1

            print('Second run. Decrease weights bitsize')
            while 1:
                weight_bit_precision -= 1
                bias_bit_precision = image_bit_precision
                print('\nStart image bit precision: {} Weights precision: {} Bias precision: {}'.format(image_bit_precision, weight_bit_precision, bias_bit_precision))
                error_rate, pred_float, pred_fixed = go_mat_model(model, images, image_bit_precision, weight_bit_precision, bias_bit_precision, debug_info=True)
                print('Error rate: {:.6f}'.format(error_rate))
                print(res_keras_array)
                print(pred_float)
                print(pred_fixed)
                if error_rate > acceptable_error_rate:
                    weight_bit_precision += 1
                    break

            print('Third run. Decrease bias bitsize')
            while 1:
                bias_bit_precision -= 1
                print('\nStart image bit precision: {} Weights precision: {} Bias precision: {}'.format(image_bit_precision,
                                                                                                    weight_bit_precision,
                                                                                                    bias_bit_precision))
                error_rate, pred_float, pred_fixed = go_mat_model(model, images, image_bit_precision,
                                                                  weight_bit_precision, bias_bit_precision,
                                                                  debug_info=True)
                print('Error rate: {:.6f}'.format(error_rate))
                print(res_keras_array)
                print(pred_float)
                print(pred_fixed)
                if error_rate > acceptable_error_rate:
                    bias_bit_precision += 1
                    break

        if 0:
            print('Single debug run')
            print('\nStart error precision: {} Weights precision: {} Bias precision: {}'.format(image_bit_precision,
                                                                                                weight_bit_precision,
                                                                                                bias_bit_precision))
            error_rate, pred_float, pred_fixed = go_mat_model(model, images, image_bit_precision,
                                                              weight_bit_precision, bias_bit_precision,
                                                              debug_info=True)
            print('Error rate: {:.6f}'.format(error_rate))
            print(res_keras_array)
            print(pred_float)
            print(pred_fixed)

        save_in_file((image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB), cache_path)
        return image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB
    else:
        return load_from_file(cache_path)


if __name__ == '__main__':
    if 0:
        use_cache = False
        acceptable_error_rate = 0.005 # 0.5%
        image_limit = 3000
        type = 'people'
        model_path_rescaled = MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_people_loss_0.3600_acc_0.8442_epoch_38_reduced_rescaled.h5'

    if 0:
        use_cache = False
        acceptable_error_rate = 0.005  # 0.5%
        image_limit = 3000
        type = 'cars'
        model_path_rescaled = MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_cars_loss_0.1088_acc_0.9631_epoch_67_reduced_rescaled.h5'

    if 1:
        use_cache = False
        acceptable_error_rate = 0.005  # 0.5%
        image_limit = 3000
        type = 'animals'
        model_path_rescaled = MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33_reduced_rescaled.h5'


    image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB = get_optimal_bit_for_weights(type, model_path_rescaled, image_limit, acceptable_error_rate, use_cache)
    if image_bit_precision > 0:
        print('Optimal bit size for image and feature maps (sign bit is not included) is: {}'.format(image_bit_precision))
        print('Optimal bit size for weights: {}'.format(weight_bit_precision))
        print('Optimal bit size for bias: {}'.format(bias_bit_precision))
        print('Bit overflows. Weights {} Bias: {}'.format(convW, convB))
    else:
        print('Impossible to find optimal bit!')
    sess.close()

'''
Max error rate: 0.5%
weights_mobilenet_1_0.25_128px_people_loss_0.3600_acc_0.8442_epoch_38_reduced_rescaled.h5
Optimal 12, 11, 10, 7, 3
weights_mobilenet_1_0.25_128px_cars_loss_0.1088_acc_0.9631_epoch_67_reduced_rescaled.h5
Optimal 10, 9, 8, 7, 3
weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33_reduced_rescaled.h5

'''