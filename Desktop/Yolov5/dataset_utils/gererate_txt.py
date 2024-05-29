#generage txt file that locate each training image

import os
import pandas as pd
import shutil
import linecache

data_type = 'train'

def insert(line):
    file = './'+data_type+'.txt'
    with open(file, 'a+') as f:
        print(line)
        f.write('%s\n' % line)


def main():
    #path = './images/train/1.tif' file number: 1 to 2388
    num = 1
    while num < 2389:
        number = str(num)
        filename = "%s.tif" % number
        file_path = './Culvert/images/'+data_type+'/%s' % filename
        if os.path.exists(file_path):
            line = './images/%s.tif' % number
            #insert a line of address
            insert(line)
        num = num +1

main()