import serial
from tqdm import tqdm

FILE_TO_WRITE = 'weights/weights_cars.txt'


if __name__ == '__main__':
    ser = serial.Serial(port='COM4', baudrate=115200, bytesize=8, timeout=0)
    ser.write(bytes([255]))

    file = open(FILE_TO_WRITE, 'r')
    k = 0
    j = 0
    l = 0
    for i in file:
        k += 1

    file.close()

    file = open(FILE_TO_WRITE, 'r')
    for i in tqdm(range(k)):
        string_current = file.readline()
        if ((string_current.split(" ")[0] != "\n") & (string_current.split(" ")[0] != "//")):
            if string_current.split(" ")[2] == '':
                data_current = string_current.split(" ")[3][4:-1]
                minus = '0'
            else:
                data_current = string_current.split(" ")[2][5:-1]
                minus = '1'

            while len(data_current) != 21:
                data_current = "0" + data_current

            ser.write(bytes([int('00' + data_current[15] + data_current[16] + data_current[17] + data_current[18] + data_current[19] + data_current[20], 2)]))
            ser.write(bytes([int('00' + data_current[9] + data_current[10] + data_current[11] + data_current[12] + data_current[13] + data_current[14], 2)]))
            ser.write(bytes([int('00' + data_current[3] + data_current[4] + data_current[5] + data_current[6] + data_current[7] + data_current[8], 2)]))
            ser.write(bytes([int('0000' + minus + data_current[0] + data_current[1] + data_current[2], 2)]))
            l += 1

    for i in range(3):
        ser.write(bytes([191]))
    for i in range(3):
        ser.write(bytes([0]))
    file.close()

    print("Counter numbers: " + str(l))