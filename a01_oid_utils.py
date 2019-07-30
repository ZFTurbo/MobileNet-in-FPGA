# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'

import pyvips
import platform
from PIL import Image
from a00_common_functions import *

# Paths and constants
if platform.processor() == 'Intel64 Family 6 Model 79 Stepping 1, GenuineIntel':
    DATASET_PATH = 'E:/Projects_M2/2019_06_Google_Open_Images/input/'
else:
    DATASET_PATH = 'E:/Projects_2TB/2019_06_Google_Open_Images/input/'
STORAGE_PATH_TRAIN = DATASET_PATH + 'train/'
STORAGE_PATH_TEST = DATASET_PATH + 'test/'
STORAGE_PATH_VALID = DATASET_PATH + 'validation/'


def get_description_for_labels():
    out = open(DATASET_PATH + 'class-descriptions-boxable.csv')
    lines = out.readlines()
    ret_1, ret_2 = dict(), dict()
    for l in lines:
        arr = l.strip().split(',')
        ret_1[arr[0]] = arr[1]
        ret_2[arr[1]] = arr[0]
    return ret_1, ret_2


def random_intensity_change(img, max_change):
    img = img.astype(np.float32)
    for j in range(3):
        delta = random.randint(-max_change, max_change)
        img[:, :, j] += delta
    img[img < 0] = 0
    img[img > 255] = 255
    return img


def random_rotate(image, max_angle):
    cols = image.shape[1]
    rows = image.shape[0]

    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return dst


def read_single_image(path):
    try:
        img = pyvips.Image.new_from_file(path, access='sequential')
        img = np.ndarray(buffer=img.write_to_memory(),
                         dtype=np.uint8,
                         shape=[img.height, img.width, img.bands])
    except:
        print('Pyvips error! {}'.format(path))
        try:
            img = np.array(Image.open(path))
        except:
            try:
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            except:
                print('Fail')
                return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 2:
        img = img[:, :, :1]

    if img.shape[2] == 1:
        img = np.concatenate((img, img, img), axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img


def read_image_bgr_fast(path):
    img2 = read_single_image(path)
    try:
        img2 = img2[:, :, ::-1]
    except:
        return None
    return img2


def prepare_training_csv(type, true_labels_enc, output_path, side_size=128, min_class_size=5):
    boxes = pd.read_csv(DATASET_PATH + 'annotations/{}-annotations-bbox.csv'.format(type))
    print('Initial boxes: {}'.format(len(boxes)))
    image_ids = boxes['ImageID'].unique()
    print('Unique images: {}'.format(len(image_ids)))
    boxes_part = boxes[boxes['LabelName'].isin(true_labels_enc)]
    print('Potential needed class boxes: {}'.format(len(boxes_part)))
    print('Potential images with class: {}'.format(len(boxes_part['ImageID'].unique())))

    images_with_needed_class = set()
    for index, row in boxes_part.iterrows():
        x1 = row['XMin']
        x2 = row['XMax']
        y1 = row['YMin']
        y2 = row['YMax']
        if (x2-x1)*side_size >= min_class_size and (y2-y1)*side_size >= min_class_size:
            images_with_needed_class |= {row['ImageID']}
    print('Images with class reduced: {}'.format(len(images_with_needed_class)))
    no_class = list(set(image_ids) - set(images_with_needed_class))
    print('Images without class: {}'.format(len(no_class)))

    out = open(output_path, 'w')
    out.write('id,target\n')
    for id in sorted(list(images_with_needed_class)):
        out.write(id + ',1\n')
    for id in sorted(list(no_class)):
        out.write(id + ',0\n')
    out.close()


def check_validation_set(input_csv):
    s = pd.read_csv(input_csv)

    print('Go for true')
    s_true = s[s['target'] == 1]
    ids_true = list(s_true['id'].values)
    for id in ids_true[:10]:
        img = cv2.imread(STORAGE_PATH_VALID + id + '.jpg')
        show_image(img)

    print('Go for false')
    s_true = s[s['target'] == 0]
    ids_true = list(s_true['id'].values)
    for id in ids_true[:10]:
        img = cv2.imread(STORAGE_PATH_VALID + id + '.jpg')
        show_image(img)


def check_train_set(input_csv):
    s = pd.read_csv(input_csv)

    print('Go for true')
    s_true = s[s['target'] == 1]
    ids_true = list(s_true['id'].values)
    for id in ids_true[:10]:
        img = cv2.imread(STORAGE_PATH_TRAIN + id[:3] + '/' + id + '.jpg')
        show_image(img)

    print('Go for false')
    s_true = s[s['target'] == 0]
    ids_true = list(s_true['id'].values)
    for id in ids_true[:10]:
        img = cv2.imread(STORAGE_PATH_TRAIN + id[:3] + '/' + id + '.jpg')
        show_image(img)


def get_class_labels(true_labels):
    d1, d2 = get_description_for_labels()
    arr = []
    for t in true_labels:
        arr.append(d2[t])
    print(arr)
    return arr
