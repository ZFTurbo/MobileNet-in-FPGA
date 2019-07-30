# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

'''
Prepare dataset for cars from Open Images Dataset (OID) from google
1) Car must be at least 5px size (for 128x128 image)
'''


if __name__ == '__main__':
    import os
    gpu_use = 0
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a01_oid_utils import *

# Definition for people
TRUE_LABELS_PEOPLE = ['Person', 'Man', 'Woman', 'Boy', 'Girl', 'Human body', 'Human eye', 'Skull', 'Human head', 'Human face',
          'Human mouth', 'Human ear', 'Human nose', 'Human hair', 'Human hand', 'Human foot', 'Human arm', 'Human leg',
          'Human beard']
TRUE_LABELS_PEOPLE_ENC = ['/m/01g317', '/m/04yx4', '/m/03bt1vf', '/m/01bl7v', '/m/05r655', '/m/02p0tk3',
                   '/m/014sv8', '/m/016m2d', '/m/04hgtk', '/m/0dzct', '/m/0283dt1', '/m/039xj_',
                   '/m/0k0pj', '/m/03q69', '/m/0k65p', '/m/031n1', '/m/0dzf4', '/m/035r7c', '/m/015h_t']

# Definition for cars
TRUE_LABELS_CAR = ['Car', 'Van', 'Taxi', 'Limousine', 'Truck', 'Bus', 'Ambulance']
TRUE_LABELS_CAR_ENC = ['/m/0k4j', '/m/0h2r6', '/m/0pg52', '/m/01lcw4', '/m/07r04', '/m/01bjv', '/m/012n7d']

SIDE_SIZE = 128
MIN_CLASS_SIZE = 5


if __name__ == '__main__':
    # Prepare people CSV
    prepare_training_csv('validation', TRUE_LABELS_PEOPLE_ENC, CACHE_PATH + 'oid_validation_people.csv', SIDE_SIZE, MIN_CLASS_SIZE)
    prepare_training_csv('train', TRUE_LABELS_PEOPLE_ENC, CACHE_PATH + 'oid_train_people.csv', SIDE_SIZE, MIN_CLASS_SIZE)

    # Prepare cars CSV
    prepare_training_csv('validation', TRUE_LABELS_CAR_ENC, CACHE_PATH + 'oid_validation_cars.csv', SIDE_SIZE, MIN_CLASS_SIZE)
    prepare_training_csv('train', TRUE_LABELS_CAR_ENC, CACHE_PATH + 'oid_train_cars.csv', SIDE_SIZE, MIN_CLASS_SIZE)


'''
Initial boxes: 204621
Unique images: 35925
Potential car boxes: 10070
Potential images with car: 5204
Images with car reduced: 5200
Images without car: 30725

Initial boxes: 14610229
Unique images: 1743042
Potential car boxes: 284869
Potential images with car: 104864
Images with car reduced: 104517
Images without car: 1638525
'''