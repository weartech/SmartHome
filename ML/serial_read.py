import serial
from nltk import word_tokenize
import csv
import time

serial_port = '/dev/ttyACM0';
baud_rate = 9600; #In arduino, Serial.begin(baud_rate)
label = "arm swing"
infile = "./data/raw/" + label
time_duration = 30

output_file = open(infile, "w+");
ser = serial.Serial(serial_port, baud_rate)
start_time = time.time()
while time.time() - start_time < time_duration:
    line = ser.readline();
    line = line.decode("utf-8") #ser.readline returns a binary, convert to string
    print(line);
    output_file.write(line);

infile = './data/raw/output.txt'
outfile = "./data/" + label + ".csv"

#with open(outfile , 'w+') as out:
    #out.write('ax,ay,az,gx,gy,gz,mx,my,mz,q0,qx,qy,qz,Yaw,Pitch,Roll,Rate\n')

with open(infile, 'r') as file:
    for index , row in enumerate(file):
        line_index = index % 6 # every 5 lines

        line_tokens = word_tokenize(row)
        
        if line_index == 0:
            line_tuple = tuple()
        
        if line_index in [0 , 1 , 2]:
            line_tuple += float(line_tokens[2]) , float(line_tokens[5]) , float(line_tokens[8])
        if line_index == 3:
            line_tuple += float(line_tokens[2]), float(line_tokens[5]) , float(line_tokens[8]) , float(line_tokens[11])
        if line_index == 4:
            line_tuple += float(line_tokens[6]) , float(line_tokens[8]) , float(line_tokens[10])
        if line_index == 5:
            line_tuple += float(line_tokens[2]), 

        if line_index == 5:
            with open(outfile,'a') as out:
                csv_out=csv.writer(out)
                csv_out.writerow(line_tuple)

            
    
