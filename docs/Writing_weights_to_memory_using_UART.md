The [UART](https://en.wikipedia.org/wiki/Universal_asynchronous_receiver-transmitter) port is used to write the neural net weights to the FPGA memory. And the weights are transferred directly from 
the PC using the Python language. First you need to connect all the necessary wires to OpenVino. See picture below.

![Wires](https://github.com/ZFTurbo/MobileNet-in-FPGA/blob/master/img/FPGA-Img-01.jpg)

Next, turn on the device. Then launch Quartus Prime (we used version 18.0) and flash the entire project 
using Programmer. Next, run a special Python script: 
[data_uart_to_fpga.py](https://github.com/ZFTurbo/MobileNet-in-FPGA/blob/master/utils/data_uart_to_fpga.py). 
Make sure that your folder with weights is next to the executable file and has the correct name.
Depending on what weights you need - for people, animals or cars - select the appropriate file. 
It must be written in the code [WEIGHT_FILE_TO_USE](https://github.com/ZFTurbo/MobileNet-in-FPGA/blob/master/utils/data_uart_to_fpga.py#4). 
If everything is done correctly then progress will go.

![Wires](https://github.com/ZFTurbo/MobileNet-in-FPGA/blob/master/img/FPGA-Img-04.png)

Upon completion of loading weights, an image from the camera will appear on the screen and the 
neural network will start to recognize it. The result will be displayed in the upper left 
corner in red (there is an required object) or in green (the required object is missing).