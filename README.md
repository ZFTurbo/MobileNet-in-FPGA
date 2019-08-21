# MobileNet in FPGA
Generator of verilog description for FPGA MobileNet implementation.
There are several pre-trained models available for frequent tasks like detection of people, cars and animals.
You can train your own model easily on your dataset using code from this repository and have the same very fast detector on FPGA working in real time for your own task.


## Requirements
Python 3.*, keras 2.2.4, tensorflow, kito


## How to run
1) `python3 r01_prepare_open_images_dataset.py` - it will create training files using [Open Images Dataset (OID)](https://storage.googleapis.com/openimages/web/index.html).
2) `python3 r02_train_mobilenet.py` - run training process. Will create weights for model and output accuracy of model.
3) `python3 r03_mobilenet_v1_reduce_and_scale_model.py` - batchnorm fusion and rescale model on range (0, 1) instead of (0, 6). Returns new rescaled model

Note: You can skip part 1, 2 and 3 if you use our pretrained weight files below

4) `python3 r04_find_optimal_bit_for_weights.py` - code to find optimal bit for feature maps, weights and biases, also returns maximum overflow for weights and biases over 1.0 value.


## Pre-trained models

|  | People detector (128px) | Cars detector (128px)  | Animals detector (128px) |
| --- | --- | --- | --- |
| Accuracy (%) | 84.42 | 96.31 | 89.67 |
| Init model (can be used for training and fine-tuning) | [people.h5](https://github.com/ZFTurbo/MobileNet-in-FPGA/releases/download/v1.0/weights_mobilenet_1_0.25_128px_people_loss_0.3600_acc_0.8442_epoch_38.h5) | [cars.h5](https://github.com/ZFTurbo/MobileNet-in-FPGA/releases/download/v1.0/weights_mobilenet_1_0.25_128px_cars_loss_0.1088_acc_0.9631_epoch_67.h5) | [animals.h5](https://github.com/ZFTurbo/MobileNet-in-FPGA/releases/download/v1.0/weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33.h5) |
| Reduced and rescaled model | [people.h5](https://github.com/ZFTurbo/MobileNet-in-FPGA/releases/download/v1.0/weights_mobilenet_1_0.25_128px_people_loss_0.3600_acc_0.8442_epoch_38_reduced_rescaled.h5) | [cars.h5](https://github.com/ZFTurbo/MobileNet-in-FPGA/releases/download/v1.0/weights_mobilenet_1_0.25_128px_cars_loss_0.1088_acc_0.9631_epoch_67_reduced_rescaled.h5) | [animals.h5](https://github.com/ZFTurbo/MobileNet-in-FPGA/releases/download/v1.0/weights_mobilenet_1_0.25_128px_animals_loss_0.2486_acc_0.8967_epoch_33_reduced_rescaled.h5) |
| Optimal bits found | 12, 11, 10, 7, 3 | 10, 9, 8, 7, 3 | 12, 11, 10, 7, 3 |
| Weights in verilog format | link | link | link |


## Description of method

Later