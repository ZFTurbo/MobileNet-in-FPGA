# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'


'''
This code takes one image run it through the network and store all intermediate feature maps 
in fixed point representation in separate files. Also detailed first pixel calculation is 
generated. It used later to check generated verilog on correctness.
'''

if __name__ == '__main__':
    import os

    # Block to choose backend and GPU to run
    gpu_use = 4
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from r04_find_optimal_bit_for_weights import *
import math


def convert_to_normalized_form_v2(value, precision):
    sign = 0
    ret = value
    if ret < 0:
        sign = 1
        ret = abs(ret)

    normed = ret
    #if sign == 1 and normed != 0:
        # Complement code for negative numbers
        #normed = 2**(precision) - normed
    down_binary_str = "{:b}".format(normed)
    for j in range(len(down_binary_str), precision):
        down_binary_str = '0' + down_binary_str
    return sign, down_binary_str


def store_layer_result(level_id, layer, layer_type, bp, res):
    r = str(res.shape[1:])[1:-1]
    r = r.replace(',', '')
    r = r.replace(' ', '_')
    out_file = INTERMEDIATE_OUTPUT_PATH + 'level_{:02d}_name_{}_bp_{}_shape_{}.txt'.format(level_id, layer.name, bp+1, r)
    print('Write to {}'.format(out_file))
    out = open(out_file, 'w')

    if layer_type != 'Conv2D' and layer_type != 'DepthwiseConv2D':
        if np.abs(res).max() >= 2 ** bp:
            print('Some layer result problem here! ({} > {})'.format(np.abs(res).max(), 2 ** bp))
            exit()
    else:
        if np.abs(res).max() >= 2 ** bp:
            print('Overflow on {} layer! ({} > {}). It is expected, increase bit space!'.format(layer_type, np.abs(res).max(), 2 ** bp))

    precision = bp + 1
    # Possible overflow. It's fine
    bit_max = math.ceil(math.log(np.abs(res).max() + 1, 2)) + 1
    if bit_max > precision:
        precision = bit_max

    total = 0
    if len(res.shape) == 4:
        if 1:
            # Start from channels
            for i in range(res.shape[3]):
                for j in range(res.shape[1]):
                    for k in range(res.shape[2]):
                        sign, bin1 = convert_to_normalized_form_v2(res[0, j, k, i].copy(), precision)
                        sgn = ' '
                        if sign == 1:
                            sgn = '-'
                        out.write("pixel[{}] = {}{}'b{}; // {}\n".format(total, sgn, precision, bin1, res[0, j, k, i]))
                        total += 1
                out.write('\n')
    elif len(res.shape) == 2:
        for i in range(res.shape[1]):
            sign, bin1 = convert_to_normalized_form_v2(res[0, i].copy(), precision)
            sgn = ' '
            if sign == 1:
                sgn = '-'
            out.write("pixel[{}] = {}{}'b{}; // {}\n".format(total, sgn, precision, bin1, res[0, i]))
            total += 1
    else:
        print('Shape problem!')
        exit()

    out.close()


def print_convolution_detailed_first_pixel_calculation(level_id, layer, img, image_bit_precizion, weight_bit_precision, bias_bit_precision):
    config = layer.get_config()
    filters = config['filters']
    use_bias = config['use_bias']
    strides = config['strides']
    padding = config['padding']
    kernel_size = config['kernel_size']

    sh1 = img.shape[1]
    sh2 = img.shape[2]
    if padding == 'valid':
        sh1 -= 2
        sh2 -= 2

    if kernel_size != (3, 3) and kernel_size != (1, 1):
        print('Unsupported kernel size: {}'.format(kernel_size))
        exit()

    (w, b) = layer.get_weights()

    out_file = FIRST_PIXEL_OUTPUT_PATH + 'level_{:02d}_name_{}_bp_{}.txt'.format(level_id, layer.name, image_bit_precizion + 1)
    print('Write to {}'.format(out_file))
    out = open(out_file, 'w')
    w = convert_to_fix_point(w.copy(), weight_bit_precision)
    b = convert_to_fix_point(b.copy(), bias_bit_precision)

    i = 0
    x = 0
    y = 0
    # Output filter number
    wi = i
    # Batch image number
    sh0 = 0
    out.write('Point: {} X: {} Y: {}\n'.format(i, x, y))

    # input filters cycle
    value = 0
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

        # Convolution for single output pixel
        i = x*strides[0] + 1
        j = y*strides[1] + 1
        if kernel_size == (3, 3):
            sub = zero_pad[i - 1:i + 2, j - 1:j + 2]
            vv = np.sum((sub * kernel).astype(np.int64))
        elif kernel_size == (1, 1):
            sub = zero_pad[i, j]
            vv = (sub * kernel[0, 0]).astype(np.int64)

        out.write('Input kernel number: {}\n'.format(wj))
        out.write('Kernel:\n{}\n'.format(kernel))
        out.write('Part:\n{}\n'.format(sub))
        out.write('Current result: {}\n\n'.format(vv))
        value += vv

    b[wi] <<= weight_bit_precision + (image_bit_precizion - bias_bit_precision)
    out.write('Add bias: {}\n'.format(b[wi]))
    value += b[wi]
    out.write('Overall result before shift: {}\n'.format(value))
    # Divide by 2^bp
    value = np.right_shift(value, weight_bit_precision)
    out.write('Overall result after shift: {}\n'.format(value))
    out.close()


def print_depthwise_conv_detailed_first_pixel_calculation(level_id, layer, img, image_bit_precizion, weight_bit_precision, bias_bit_precision):
    config = layer.get_config()
    use_bias = config['use_bias']
    strides = config['strides']
    padding = config['padding']
    kernel_size = config['kernel_size']

    sh1 = img.shape[1]
    sh2 = img.shape[2]
    if padding == 'valid':
        sh1 -= 2
        sh2 -= 2

    if kernel_size != (3, 3):
        print('Unsupported kernel size: {}'.format(kernel_size))
        exit()

    (w, b) = layer.get_weights()

    out_file = FIRST_PIXEL_OUTPUT_PATH + 'level_{:02d}_name_{}_bp_{}.txt'.format(level_id, layer.name, image_bit_precizion+1)
    print('Write to {}'.format(out_file))
    out = open(out_file, 'w')
    w = convert_to_fix_point(w.copy(), weight_bit_precision)
    b = convert_to_fix_point(b.copy(), bias_bit_precision)

    i = 0
    x = 0
    y = 0
    # Output filter number
    wj = i
    wi = 0
    # Batch image number
    sh0 = 0
    out.write('Point: {} X: {} Y: {}\n'.format(i, x, y))

    # input filters cycle
    value = 0
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

    # Convolution for single output pixel
    i = x*strides[0] + 1
    j = y*strides[1] + 1
    if kernel_size == (3, 3):
        sub = zero_pad[i - 1:i + 2, j - 1:j + 2]
        vv = np.sum((sub * kernel).astype(np.int64))
    elif kernel_size == (1, 1):
        sub = zero_pad[i, j]
        vv = (sub * kernel[0, 0]).astype(np.int64)

    value = vv
    out.write('Input kernel number: {}\n'.format(wj))
    out.write('Kernel:\n{}\n'.format(kernel))
    out.write('Part:\n{}\n'.format(sub))
    b[wj] <<= weight_bit_precision + (image_bit_precizion - bias_bit_precision)
    out.write('Add bias: {}\n'.format(b[wj]))
    value += b[wj]
    out.write('Overall result before shift: {}\n'.format(value))
    # Divide by 2^bp
    value = np.right_shift(value, weight_bit_precision)
    out.write('Overall result after shift: {}\n'.format(value))
    out.close()


def print_dense_detailed_first_pixel_calculation(level_id, layer, img, image_bit_precizion, weight_bit_precision):
    config = layer.get_config()
    use_bias = config['use_bias']
    activation = config['activation']

    if use_bias:
        (w, b) = layer.get_weights()
    else:
        (w,) = layer.get_weights()

    if use_bias is True:
        print('Bias currently not supported!')
        exit()

    if activation != 'softmax':
        print('Activation {} is not supported'.format(activation))
        exit()

    w = convert_to_fix_point(w.copy(), weight_bit_precision)

    out_file = FIRST_PIXEL_OUTPUT_PATH + 'level_{:02d}_name_{}_bp_{}.txt'.format(level_id, layer.name,
                                                                                 weight_bit_precision + 1)
    print('Write to {}'.format(out_file))
    out = open(out_file, 'w')

    i = 0
    x = 0
    # Batch image number
    sh0 = 0
    out.write('Point: {} X: {}\n'.format(i, x))
    value = 0
    for j in range(w.shape[0]):
        out.write('Weight {}: {}\n'.format(j, w[j, i]))
        vv = img[sh0, j] * w[j, i]
        value += vv
        out.write('Current intermediate result: {} [Accumulate: {}]\n'.format(vv, value))

    out.write('Overall result before shift: {}\n'.format(value))
    # Divide by 2^bp
    value = np.right_shift(value.astype(np.int64), weight_bit_precision)
    out.write('Overall result after shift: {}\n'.format(value))
    out.close()


def get_filters_size(arr):
    a = np.prod(np.array(arr.shape).astype(np.int64))
    return a


def generate_layer_results(model, images, image_bit_precizion, weight_bit_precision, bias_bit_precision, convW, convB):

    if images.shape[0] > 1:
        print('Only one image must be in batch for debug!')
        exit()

    level_out_reduced = dict()
    debug_info = False
    prev_filters_space = -1
    next_filters_space = -1
    max_filter_space = -1
    critical_layer = -1

    for level_id in range(len(model.layers)):
        layer = model.layers[level_id]
        layer_type = layer.__class__.__name__
        print('Layer num: {} Layer name: {} Layer type: {}'.format(level_id, layer.name, layer_type))
        if level_id == 0:
            next_filters_space = get_filters_size(images[0]) * (image_bit_precizion + 1)

        if level_id > 0:
            print('Input shape: {}'.format(level_out_reduced[level_id-1].shape))
            prev_filters_space = next_filters_space.copy()
            next_filters_space = get_filters_size(level_out_reduced[level_id-1][0]) * (image_bit_precizion + 1)
            if prev_filters_space + next_filters_space > max_filter_space:
                max_filter_space = prev_filters_space + next_filters_space
                critical_layer = level_id

        if layer_type == 'InputLayer':
            level_out_reduced[level_id] = convert_to_fix_point(images.copy(), image_bit_precizion)
            store_layer_result(level_id, layer, layer_type, image_bit_precizion, level_out_reduced[level_id])

        elif layer_type == 'ZeroPadding2D':
            level_out_reduced[level_id] = mmZeroPadding2D_fixed_point(layer, level_out_reduced[level_id - 1].copy())

        elif layer_type == 'Conv2D':
            level_out_reduced[level_id] = mmConv2D_fixed_point(layer, level_out_reduced[level_id - 1].copy(), image_bit_precizion, weight_bit_precision, bias_bit_precision, debug_info)
            print_convolution_detailed_first_pixel_calculation(level_id, layer, level_out_reduced[level_id - 1].copy(), image_bit_precizion, weight_bit_precision, bias_bit_precision)
            store_layer_result(level_id, layer, layer_type, image_bit_precizion, level_out_reduced[level_id])

        elif layer_type == 'DepthwiseConv2D':
            level_out_reduced[level_id] = mmDepthwiseConv2D_fixed_point(layer, level_out_reduced[level_id - 1].copy(), image_bit_precizion, weight_bit_precision, bias_bit_precision, debug_info)
            print_depthwise_conv_detailed_first_pixel_calculation(level_id, layer, level_out_reduced[level_id - 1].copy(), image_bit_precizion, weight_bit_precision, bias_bit_precision)
            store_layer_result(level_id, layer, layer_type, image_bit_precizion, level_out_reduced[level_id])

        elif layer_type == 'Activation':
            level_out_reduced[level_id] = mmActivation_fixed_point(layer, level_out_reduced[level_id - 1].copy(), image_bit_precizion, debug_info)
            store_layer_result(level_id, layer, layer_type, image_bit_precizion, level_out_reduced[level_id])

        elif layer_type == 'ReLU':
            level_out_reduced[level_id] = mmReLU_fixed_point(layer, level_out_reduced[level_id - 1].copy(), image_bit_precizion, debug_info)
            store_layer_result(level_id, layer, layer_type, image_bit_precizion, level_out_reduced[level_id])

        elif layer_type == 'GlobalAveragePooling2D':
            level_out_reduced[level_id] = mmGlobalAveragePooling2D_fixed_point(level_out_reduced[level_id - 1].copy())
            store_layer_result(level_id, layer, layer_type, image_bit_precizion, level_out_reduced[level_id])

        elif layer_type == 'Dense':
            level_out_reduced[level_id] = mmDense_fixed_point(layer, level_out_reduced[level_id - 1].copy(),  image_bit_precizion, weight_bit_precision, debug_info)
            print_dense_detailed_first_pixel_calculation(level_id, layer, level_out_reduced[level_id - 1].copy(), image_bit_precizion, weight_bit_precision)
            store_layer_result(level_id, layer, layer_type, image_bit_precizion, level_out_reduced[level_id])

    print('Required space to store intermediate results of calculations: {} bits ({:.2f} MB)'.format(max_filter_space, max_filter_space / (1024 * 1024)))
    print('Critical layer number: {}'.format(critical_layer))


def get_debug_image():
    img = cv2.imread(CACHE_PATH + 'image.png')
    img_list = []
    img_list.append(img.copy())
    img_list = np.array(img_list, dtype=np.float32)
    img_list = preproc_input_mathmodel(img_list)
    print(img_list.shape, img_list.max(), img_list.min())
    return img_list


def generate_layer_results_for_image(type, model, image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB):
    print(model.summary())

    # Get only one image
    try:
        a = 10/0
        # If OID dataset exists
        images, answers = get_image_set(type, 2, 'math')
        images = images[0:1]
        print('Use OID images')
    except:
        images = np.zeros((1, 128, 128, 3), dtype=np.float32)
        images[...] = 255
        images = preproc_input_mathmodel(images)
        print('No OID images found. Use generated image')

    generate_layer_results(model, images, image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB)


if __name__ == '__main__':
    problem_type = 'people'

    INTERMEDIATE_OUTPUT_PATH = CACHE_PATH + 'intermediate_{}/'.format(problem_type)
    if not os.path.isdir(INTERMEDIATE_OUTPUT_PATH):
        os.mkdir(INTERMEDIATE_OUTPUT_PATH)
    FIRST_PIXEL_OUTPUT_PATH = CACHE_PATH + 'first_pixel_{}/'.format(problem_type)
    if not os.path.isdir(FIRST_PIXEL_OUTPUT_PATH):
        os.mkdir(FIRST_PIXEL_OUTPUT_PATH)

    if problem_type == 'people':
        model = get_model(MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_people_loss_0.3600_acc_0.8442_epoch_38_reduced_rescaled.h5')
        # bit_precision - without sign, so we need to add 1 to it to store sign as well
        # image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB = get_optimal_bit_for_weights()
        image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB = 12, 11, 10, 7, 3
    elif problem_type == 'cars':
        model = get_model(MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_cars_loss_0.1088_acc_0.9631_epoch_67_reduced_rescaled.h5')
        # bit_precision - without sign, so we need to add 1 to it to store sign as well
        # image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB = get_optimal_bit_for_weights()
        image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB = 10, 9, 8, 7, 3
    elif problem_type == 'animals':
        model = get_model(MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33_reduced_rescaled.h5')
        # bit_precision - without sign, so we need to add 1 to it to store sign as well
        # image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB = get_optimal_bit_for_weights()
        image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB = 12, 11, 10, 7, 3

    generate_layer_results_for_image(problem_type, model, image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB)
