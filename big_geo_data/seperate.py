


import os
import pandas as pd
import shutil

def seperate_files():
    IMAGE_DIR = './data_/WFBB_NE/'

    def seperate_images(file_path,source,destination,validation=False):
        file_list = pd.read_csv(file_path,dtype=str,names=['file'])
        print(file_list)
        file_list = file_list['file'].tolist()

        if not os.path.exists(destination):
            os.makedirs(destination)

        if(validation == True):
            if not os.path.exists(destination+'_xml'):
                os.makedirs(destination+'_xml')

        for filename in os.listdir(source):
            #print(filename)
            current_file = os.path.splitext(filename)[0]


            if current_file in file_list:
                filename = os.path.join(source, filename)
                annotation_file = os.path.join('./annotations_orig_trainData', current_file + '.xml')
                print(annotation_file)
                os.path.join(source, filename)
                print("Copying" + filename)
                shutil.copy(filename, destination)
                if validation == False:
                    shutil.copy(annotation_file, destination)
                else:
                    shutil.copy(annotation_file, destination+'_xml')


            


    seperate_images("ImageSets/train.txt",IMAGE_DIR,'./train_new')
    seperate_images("ImageSets/test.txt",IMAGE_DIR,'./test_new')
    seperate_images("ImageSets/trainval.txt",IMAGE_DIR,'./trainval_new')
    seperate_images("ImageSets/val.txt",IMAGE_DIR,'./val_new',validation=True)


if __name__ == '__main__':
   seperate_files()
