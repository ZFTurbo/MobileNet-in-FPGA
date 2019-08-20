# MobileNet in FPGA
Generator of verilog description for FPGA MobileNet implementation


## Requirements
Python 3.*, keras 2.2.4, tensorflow, kito


## How to run
1) `python3 r01_prepare_open_images_dataset.py` - it will create training files using Open Images Dataset (OID).
2) `python3 r02_train_mobilenet.py` - run training process. Will create weights for model and output accuracy of model.
3) `python3 r03_mobilenet_v1_reduce_and_scale_model.py` - batchnorm fusion and rescale model on range (0, 1) instead of (0, 6). Returns new rescaled model

Note: You can skip part 1, 2 and 3 if you use our pretrained weight files below

4) `python3 r04_find_optimal_bit_for_weights.py` - code to find optimal bit for feature maps, weights and biases, also returns maximum overflow for weights and biases over 1.0 value.


## Pre-trained models

|  | People detector (128px) | Cars detector (128px)  | Animals detector (128px) |
| --- | --- | --- | --- |
| Accuracy (%) | 84.42 | 96.31 | 89.67 |
| Init model (can be used for training and fine-tuning) | people.h5 | cars.h5 | [animals.h5](https://github.com/ZFTurbo/MobileNet-in-FPGA/releases/download/v1.0/weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33.h5) |
| Reduced and rescaled model | people.h5 | cars.h5 | animals.h5 |
| Optimal bits found | 12, 11, 10, 7, 3 | 10, 9, 8, 7, 3 | 12, 11, 10, 7, 3 |
| Weights in verilog format | link | link | link |

## Description of method

Later