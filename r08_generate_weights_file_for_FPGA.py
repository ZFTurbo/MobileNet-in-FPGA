# coding: utf-8
__author__ = 'Alex Kustov, IPPM RAS'

def load_cache_file(f):
    file = open(f,'r')
    result_list = []
    for i in file:
        if ((i[:2] != '//') & (i != '\n')):
            result_list.append(i)
    file.close()
    return result_list

if __name__ == '__main__':
    nn_type = 'people'

    f_w = 'cache/{}/storage.v'.format(nn_type)
    f_b = 'cache/{}/storage_bias.v'.format(nn_type)
    f_r = 'cache/{}/weights.txt'.format(nn_type)

    weights = load_cache_file(f_w)
    bias = load_cache_file(f_b)

    file = open(f_r,'w')
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            file.write(weights[i][j])
    for i in range(3):
        file.write('storage[0] =  12\'b000000000000; // 0 0\n')
    for i in range(len(bias)):
        for j in range(len(bias[i])):
            file.write(bias[i][j])
    file.close()