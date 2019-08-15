# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

'''
Prepare dataset for different classes from Open Images Dataset (OID) from google
1) Class must be at least 5px size (for 128x128 image)
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

# Definition for animals
TRUE_LABELS_ANIMAL = ['Animal', 'Bird', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin', 'Raven', 'Chicken',
                      'Eagle', 'Owl', 'Duck', 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow',
                      'Turkey', 'Invertebrate', 'Tick', 'Centipede', 'Marine invertebrates', 'Starfish',
                      'Lobster', 'Jellyfish', 'Shrimp', 'Crab', 'Insect', 'Bee', 'Beetle', 'Ladybug',
                      'Ant', 'Moths and butterflies', 'Caterpillar', 'Butterfly', 'Dragonfly', 'Spider',
                      'Oyster', 'Snail', 'Bat', 'Carnivore', 'Bear', 'Brown bear', 'Polar bear', 'Cat',
                      'Fox', 'Jaguar', 'Lynx', 'Tiger', 'Lion', 'Dog', 'Leopard', 'Cheetah', 'Otter',
                      'Raccoon', 'Camel', 'Cattle', 'Giraffe', 'Rhinoceros', 'Goat', 'Horse', 'Hamster',
                      'Kangaroo', 'Mouse', 'Pig', 'Rabbit', 'Squirrel', 'Sheep', 'Zebra', 'Monkey', 'Deer',
                      'Elephant', 'Porcupine', 'Bull', 'Antelope', 'Mule', 'Marine mammal', 'Dolphin', 'Whale',
                      'Sea lion', 'Harbor seal', 'Alpaca', 'Reptile', 'Dinosaur', 'Lizard', 'Snake', 'Turtle',
                      'Tortoise', 'Sea turtle', 'Crocodile', 'Frog', 'Fish', 'Goldfish', 'Shark', 'Seahorse',
                      'Shellfish']

TRUE_LABELS_ANIMAL_ENC = ['/m/0jbk', '/m/015p6', '/m/01dy8n', '/m/01f8m5', '/m/05n4y', '/m/05z6w', '/m/06j2d',
                          '/m/09b5t', '/m/09csl', '/m/09d5_', '/m/09ddx', '/m/0ccs93', '/m/0dbvp', '/m/0dftk',
                          '/m/0f6wt', '/m/0gv1x', '/m/0h23m', '/m/0jly1', '/m/03xxp', '/m/0175cv', '/m/019h78',
                          '/m/03hl4l9', '/m/01h8tj', '/m/0cjq5', '/m/0d8zb', '/m/0ll1f78', '/m/0n28_', '/m/03vt0',
                          '/m/01h3n', '/m/020jm', '/m/0gj37', '/m/0_k2', '/m/0d_2m', '/m/0cydv', '/m/0cyf8',
                          '/m/0ft9s', '/m/09kmb', '/m/0_cp5', '/m/0f9_l', '/m/01h44', '/m/01lrl', '/m/01dws',
                          '/m/01dxs', '/m/0633h', '/m/01yrx', '/m/0306r', '/m/0449p', '/m/04g2r', '/m/07dm6',
                          '/m/096mb', '/m/0bt9lr', '/m/0c29q', '/m/0cd4d', '/m/0cn6p', '/m/0dq75', '/m/01x_v',
                          '/m/01xq0k1', '/m/03bk1', '/m/03d443', '/m/03fwl', '/m/03k3r', '/m/03qrc', '/m/04c0y',
                          '/m/04rmv', '/m/068zj', '/m/06mf6', '/m/071qp', '/m/07bgp', '/m/0898b', '/m/08pbxl',
                          '/m/09kx5', '/m/0bwd_0j', '/m/0c568', '/m/0cnyhnx', '/m/0czz2', '/m/0dbzx', '/m/0gd2v',
                          '/m/02hj4', '/m/084zz', '/m/0gd36', '/m/02l8p9', '/m/0pcr', '/m/06bt6', '/m/029tx',
                          '/m/04m9y', '/m/078jl', '/m/09dzg', '/m/011k07', '/m/0120dh', '/m/09f_2', '/m/09ld4',
                          '/m/0ch_cf', '/m/03fj2', '/m/0by6g', '/m/0nybt', '/m/0fbdv']

SIDE_SIZE = 128
MIN_CLASS_SIZE = 5


if __name__ == '__main__':
    # Prepare people CSV
    prepare_training_csv('validation', TRUE_LABELS_PEOPLE_ENC, CACHE_PATH + 'oid_validation_people.csv', SIDE_SIZE, MIN_CLASS_SIZE)
    prepare_training_csv('train', TRUE_LABELS_PEOPLE_ENC, CACHE_PATH + 'oid_train_people.csv', SIDE_SIZE, MIN_CLASS_SIZE)

    # Prepare cars CSV
    prepare_training_csv('validation', TRUE_LABELS_CAR_ENC, CACHE_PATH + 'oid_validation_cars.csv', SIDE_SIZE, MIN_CLASS_SIZE)
    prepare_training_csv('train', TRUE_LABELS_CAR_ENC, CACHE_PATH + 'oid_train_cars.csv', SIDE_SIZE, MIN_CLASS_SIZE)

    # Prepare animals CSV
    prepare_training_csv('validation', TRUE_LABELS_ANIMAL_ENC, CACHE_PATH + 'oid_validation_animals.csv', SIDE_SIZE, MIN_CLASS_SIZE)
    prepare_training_csv('train', TRUE_LABELS_ANIMAL_ENC, CACHE_PATH + 'oid_train_animals.csv', SIDE_SIZE, MIN_CLASS_SIZE)
