# MobileNet in FPGA
Generator of verilog description for FPGA MobileNet implementation


## Requirements
Python 3.*, keras 2.2.4, tensorflow, kito


## How to run
1) `python3 r01_prepare_open_images_dataset.py` - it will create training files using Open Images Dataset (OID).
2) `python3 r02_train_mobilenet.py` - run training process. Will create weights for model and output accuracy of model.
3) `python3 r03_remove_batchnorm_layers.py` - batchnorm fusion. Returns new model file without BatchNormalization layers

Note: You can skip part 1, 2 and 3 if you use our pretrained weight files: (link)

4)