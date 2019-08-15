# MobileNet in FPGA
Generator of verilog description for FPGA MobileNet implementation

## How to run

`python3 r01_prepare_open_images_dataset.py` - it will create training files using Open Images Dataset (OID).
`python3 r02_train_mobilenet.py` - run training process. Will create weights for model and output accuracy of model.

You can skip this part if you use our pretrained weight files: (link)

