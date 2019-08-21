# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

'''
Generate weights with optimal bit size in verilog format 
'''

if __name__ == '__main__':
    import os

    # Block to choose backend
    gpu_use = 4
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from r04_find_optimal_bit_for_weights import *


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
        print('Avg difference: {}'.format(s.mean()/size))
        print('Value range float: {} - {}'.format(s1.min(), s1.max()))
        print('Value range fixed: {} - {}'.format(s2.min(), s2.max()))


def dump_memory_structure_conv(arr, out_file):
    print('Dump memory structure in file: {}'.format(out_file))
    out = open(out_file, "w")
    total = 0
    for a in range(arr.shape[2]):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out.write(str(total) + " LVL: {} X: {} Y: {} ".format(a, i, j) + str(arr[i, j, a]) + '\n')
                total += 1

    out.close()


def dump_memory_structure_dense(arr, out_file):
    print('Dump memory structure for dense layer in file: {}'.format(out_file))
    out = open(out_file, "w")
    total = 0
    print('Shape:', arr.shape)
    for j in range(arr.shape[0]):
         out.write(str(total) + " POS: {} ".format(j) + str(arr[j]) + '\n')
         total += 1

    out.close()


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


def convert_to_normalized_form(value, precision, required_precision=None):
    sign = 0
    ret = value
    if ret < 0:
        sign = 1
        ret = abs(ret)

    # down = ret - math.floor(ret)
    # print(ret, down)
    normed = int(round(ret * 2**(precision-1)))
    #if sign == 1 and normed != 0:
        # Complement code for negative numbers
        #normed = 2**(precision) - normed
    down_binary_str = "{:0b}".format(normed)
    if required_precision is None:
        required_precision = precision
    for j in range(len(down_binary_str), required_precision):
        down_binary_str = '0' + down_binary_str
    return sign, down_binary_str


def convert_to_normalized_form_array(value, precision):
    ret = np.abs(value)
    normed = np.round(ret * 2**(precision - 1)).astype(np.int64)
    return normed


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


def get_shape_string(w):
    r = str(w.shape)[1:-1]
    r = r.replace(',', '')
    r = r.replace(' ', '_')
    return r


def gen_convolution_weights(level_id, layer, bit_precizion, weight_bit_precision, bias_bit_precision, convW, convB, out_dir):
    # Convolution with fixed point
    config = layer.get_config()
    use_bias = config['use_bias']
    kernel_size = config['kernel_size']
    requred_mem_in_bits = 0

    if kernel_size != (3, 3) and kernel_size != (1, 1):
        print('Unsupported kernel size: {}'.format(kernel_size))
        exit()

    (w, b) = layer.get_weights()
    # w = convert_to_fix_point(w.copy(), bit_precizion)

    # Check that everything is fine with weights
    if w[w > 2 ** convW].any() or w[w < -2 ** convW].any():
        print('Overflow for conv weights!')
        exit()

    # Check that everything is fine with bias
    if b[b > 2 ** convB].any() or b[b < -2 ** convB].any():
        print('Overflow for conv bias!')
        exit()

    precisionW = weight_bit_precision + 1 + convW
    precisionB = bias_bit_precision + 1 + convB
    print('Initial bits weights: {} bias: {}'.format(precisionW, precisionB))

    w_check = convert_to_normalized_form_array(w, weight_bit_precision + 1)
    w_check_max = w_check.max()
    precisionW = np.log2(w_check_max).astype(np.int64) + 1 + 1
    b_check = convert_to_normalized_form_array(b, bias_bit_precision + 1)
    b_check_max = b_check.max()
    precisionB = np.log2(b_check_max).astype(np.int64) + 1 + 1
    print('Max value to store weights: {} bias: {}'.format(w_check_max, b_check_max))
    print('Reduced bits weights: {} bias: {}'.format(precisionW, precisionB))

    out_file = out_dir + 'level_{:02d}_name_{}_bpset_{}_{}_{}_shape_{}.txt'.format(level_id, layer.name,
                bit_precizion + 1, weight_bit_precision + 1 + convW, bias_bit_precision + 1 + convB, get_shape_string(w))
    out = open(out_file, 'w')
    print('Go for: {} Shape: {}'.format(layer.name, w.shape))

    tp1 = 'bin'
    total = 0

    # Cycle by outputs
    for i in range(w.shape[3]):
        # Cycle by inputs
        for j in range(w.shape[2]):
            # Cycle by conv 3x3
            for k in range(w.shape[1]):
                for l in range(w.shape[0]):
                    sign, bin1 = convert_to_normalized_form(w[k, l, j, i].copy(), weight_bit_precision + 1, precisionW)
                    sgn = ' '
                    if sign == 1:
                        sgn = '-'
                    dec_verilog = int(bin1, 2)
                    if sign == 1:
                        dec_verilog = -dec_verilog
                    if tp1 == 'hex':
                        hx = hex(int(bin1, 2))[2:].upper()
                        out.write(
                            "storage[{}] = {}{}'h{}; // {} {}\n".format(total, sgn, precisionW, hx, dec_verilog, w[k, l, j, i]))
                    else:
                        out.write(
                            "storage[{}] = {}{}'b{}; // {} {}\n".format(total, sgn, precisionW, bin1, dec_verilog, w[k, l, j, i]))
                    requred_mem_in_bits += precisionW
                    total += 1
            if w.shape[1] > 1:
                out.write('\n')
        out.write('\n')

    # Cycle by outputs
    for i in range(w.shape[3]):
        sign, bin1 = convert_to_normalized_form(b[i].copy(), bias_bit_precision + 1, precisionB)
        sgn = ' '
        if sign == 1:
            sgn = '-'
        dec_verilog = int(bin1, 2)
        if sign == 1:
            dec_verilog = -dec_verilog
        if tp1 == 'hex':
            hx = hex(int(bin1, 2))[2:].upper()
            out.write(
                "storage_bias[{}] = {}{}'h{}; // {} {}\n".format(total, sgn, precisionB, hx, dec_verilog,
                                                                 b[i]))
        else:
            out.write(
                "storage_bias[{}] = {}{}'b{}; // {} {}\n".format(total, sgn, precisionB, bin1, dec_verilog,
                                                                 b[i]))
        requred_mem_in_bits += precisionB
        total += 1

    out.close()
    return requred_mem_in_bits


def gen_depthwise_convolution_weights(level_id, layer, bit_precizion, weight_bit_precision, bias_bit_precision, convW, convB, out_dir):
    config = layer.get_config()
    use_bias = config['use_bias']
    kernel_size = config['kernel_size']
    requred_mem_in_bits = 0

    if kernel_size != (3, 3) and kernel_size != (1, 1):
        print('Unsupported kernel size: {}'.format(kernel_size))
        exit()

    (w, b) = layer.get_weights()

    # Check that everything is fine with weights
    if w[w > 2 ** convW].any() or w[w < -2 ** convW].any():
        print('Overflow for conv weights!')
        exit()

    # Check that everything is fine with bias
    if b[b > 2 ** convB].any() or b[b < -2 ** convB].any():
        print('Overflow for conv bias!')
        exit()

    precisionW = weight_bit_precision + 1 + convW
    precisionB = bias_bit_precision + 1 + convB
    print('Initial bits weights: {} bias: {}'.format(precisionW, precisionB))

    w_check = convert_to_normalized_form_array(w, weight_bit_precision + 1)
    w_check_max = w_check.max()
    precisionW = np.log2(w_check_max).astype(np.int64) + 1 + 1
    b_check = convert_to_normalized_form_array(b, bias_bit_precision + 1)
    b_check_max = b_check.max()
    precisionB = np.log2(b_check_max).astype(np.int64) + 1 + 1
    print('Max value to store weights: {} bias: {}'.format(w_check_max, b_check_max))
    print('Reduced bits weights: {} bias: {}'.format(precisionW, precisionB))

    out_file = out_dir + 'level_{:02d}_name_{}_bpset_{}_{}_{}_shape_{}.txt'.format(level_id, layer.name,
                bit_precizion + 1, weight_bit_precision + 1 + convW, bias_bit_precision + 1 + convB, get_shape_string(w))
    out = open(out_file, 'w')
    print('Go for: {} Shape: {}'.format(layer.name, w.shape))

    tp1 = 'bin'
    total = 0
    # Cycle by inputs. Output is always 1
    for i in range(w.shape[2]):
        # Cycle by conv 3x3
        for k in range(w.shape[1]):
            for l in range(w.shape[0]):
                sign, bin1 = convert_to_normalized_form(w[k, l, i, 0].copy(), weight_bit_precision + 1, precisionW)
                sgn = ' '
                if sign == 1:
                    sgn = '-'
                dec_verilog = int(bin1, 2)
                if sign == 1:
                    dec_verilog = -dec_verilog
                if tp1 == 'hex':
                    hx = hex(int(bin1, 2))[2:].upper()
                    out.write(
                        "storage[{}] = {}{}'h{}; // {} {}\n".format(total, sgn, precisionW, hx, dec_verilog, w[k, l, i, 0]))
                else:
                    out.write(
                        "storage[{}] = {}{}'b{}; // {} {}\n".format(total, sgn, precisionW, bin1, dec_verilog, w[k, l, i, 0]))
                requred_mem_in_bits += precisionW
                total += 1
        out.write('\n')

    # Cycle by inputs. Output is always 1
    for i in range(w.shape[2]):
        sign, bin1 = convert_to_normalized_form(b[i].copy(), bias_bit_precision + 1, precisionB)
        sgn = ' '
        if sign == 1:
            sgn = '-'
        dec_verilog = int(bin1, 2)
        if sign == 1:
            dec_verilog = -dec_verilog
        if tp1 == 'hex':
            hx = hex(int(bin1, 2))[2:].upper()
            out.write(
                "storage_bias[{}] = {}{}'h{}; // {} {}\n".format(total, sgn, precisionB, hx, dec_verilog,
                                                            w[k, l, i, 0]))
        else:
            out.write(
                "storage_bias[{}] = {}{}'b{}; // {} {}\n".format(total, sgn, precisionB, bin1, dec_verilog,
                                                            w[k, l, i, 0]))
        requred_mem_in_bits += precisionB
        total += 1

    out.close()
    return requred_mem_in_bits


def gen_dense_weights(level_id, layer, bit_precizion, out_dir):
    config = layer.get_config()
    use_bias = config['use_bias']
    requred_mem_in_bits = 0

    if use_bias:
        print('Bias currently unsupported!')
        exit()

    (w,) = layer.get_weights()

    # Check that everything is fine with weights
    if w[w > 1].any() or w[w < -1].any():
        print('Overflow for depthwise conv weights!')
        exit()

    out_file = out_dir + 'level_{:02d}_name_{}_bp_{}_shape_{}.txt'.format(level_id, layer.name, bit_precizion,
                                                                              get_shape_string(w))
    out = open(out_file, 'w')
    print('Go for: {} Shape: {}'.format(layer.name, w.shape))

    tp1 = 'bin'
    total = 0
    precision = bit_precizion + 1
    # Cycle by outputs
    for i in range(w.shape[1]):
        # Cycle by inputs
        for j in range(w.shape[0]):
            sign, bin1 = convert_to_normalized_form(w[j, i].copy(), precision)
            sgn = ' '
            if sign == 1:
                sgn = '-'
            dec_verilog = int(bin1, 2)
            if sign == 1:
                dec_verilog = -dec_verilog
            if tp1 == 'hex':
                hx = hex(int(bin1, 2))[2:].upper()
                out.write(
                    "storage[{}] = {}{}'h{}; // {} {}\n".format(total, sgn, precision, hx, dec_verilog, w[j, i]))
            else:
                out.write(
                    "storage[{}] = {}{}'b{}; // {} {}\n".format(total, sgn, precision, bin1, dec_verilog, w[j, i]))
            total += 1
            requred_mem_in_bits += precision
        out.write('\n')

    out.close()
    return requred_mem_in_bits


def generate_weights_for_layers(model, bp, weight_bit_precision, bias_bit_precision, convW, convB, out_dir):
    weights_required_memory = 0

    for level_id in range(len(model.layers)):
        layer = model.layers[level_id]
        layer_type = layer.__class__.__name__
        req_mem = 0

        if layer_type == 'Conv2D':
            req_mem = gen_convolution_weights(level_id, layer, bp, weight_bit_precision, bias_bit_precision, convW, convB, out_dir)

        elif layer_type == 'DepthwiseConv2D':
            req_mem = gen_depthwise_convolution_weights(level_id, layer, bp, weight_bit_precision, bias_bit_precision, convW, convB, out_dir)

        elif layer_type == 'Dense':
            req_mem = gen_dense_weights(level_id, layer, weight_bit_precision, out_dir)

        else:
            continue

        print('Required weights memory: {} bit'.format(req_mem))
        weights_required_memory += req_mem

    print('Overall weights memory requirements: {} bit ({:.2f} MB)'.format(weights_required_memory, weights_required_memory / (1024*1024)))


if __name__ == '__main__':
    type = 'animals'
    model_path = MODEL_PATH + 'best/weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33_reduced_rescaled.h5'
    acceptable_error_rate = 0.005  # 0.5%
    image_limit = 3000

    model = get_model(model_path)
    if 0:
        image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB = get_optimal_bit_for_weights(type, model_path_rescaled, image_limit, acceptable_error_rate, use_cache)
    else:
        image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB = 12, 11, 10, 7, 3

    out_dir = CACHE_PATH + type + '/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    generate_weights_for_layers(model, image_bit_precision, weight_bit_precision, bias_bit_precision, convW, convB, out_dir)
