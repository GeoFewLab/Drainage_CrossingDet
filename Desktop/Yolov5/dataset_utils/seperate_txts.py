import pandas as pd
import numpy as np


train_val_split = 0.8


df = pd.read_csv('./coordinates_Bbox.csv')


#Shuffle and divide file names
file_numbers = df['Sample ID'].unique()
print(file_numbers)
print(type(file_numbers))
np.random.shuffle(file_numbers)
print(file_numbers)

split_index = np.ceil(len(file_numbers)*train_val_split)
print(split_index)

train = open('train.txt',"w")
val = open('val.txt',"w")

for i in range(len(file_numbers)):
    filename = './images/'+str(file_numbers[i])+'.tif\n'
    if(i < split_index):
        train.write(filename)
    else:
        val.write(filename)