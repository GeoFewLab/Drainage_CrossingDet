######################################################################################
### Author/Developer: Nicolas CHEN
### Filename: plotBox.py
### Version: 1.0
### Field of research: Deep Learning in medical imaging
### Purpose: This Python script plots the boxes for each image from the dataset
### Output: This Python script plots the boxes for each image and save it in 
### a new directory

######################################################################################
### HISTORY
### Version | Date          | Author       | Evolution 
### 1.0     | 17/11/2018    | Nicolas CHEN | Initial version 
######################################################################################

# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import os

def filterFiles(directoryPath, extension):
    """
        This function filters the format files with the selected extension in the directory
        
        Args:
            directoryPath (str): relative path of the directory that contains text files
            extension (str): extension file

        Returns:
            The list of filtered files with the selected extension
    """    
    relevant_path = directoryPath
    included_extensions = [extension]
    file_names = [file1 for file1 in os.listdir(relevant_path) if any(file1.endswith(ext) for ext in included_extensions)]
    numberOfFiles = len(file_names)
    listParams = [file_names, numberOfFiles]
    return listParams

[image_names, numberOfFiles] = filterFiles("Images", "tif")    

trainRCNN = pd.read_csv('test.csv', sep=",", header=None)
trainRCNN.columns = ['filename', 'box_type', 'xmin', 'xmax', 'ymin', 'ymax']


for imageFileName in image_names:    

    fig = plt.figure()
    #add axes to the image
    ax = fig.add_axes([0,0,1,1]) #adding X and Y axes from 0 to 1 for each direction 
    plt.axis('off')

    # read and plot the image
    image = plt.imread('Images/' + imageFileName)
    plt.imshow(image)
    # iterating over the image for different objects
    for _,row in trainRCNN[trainRCNN.filename == imageFileName].iterrows():
        xmin = float(row.xmin)
        xmax = float(row.xmax)
        ymin = float(row.ymin)
        ymax = float(row.ymax)

        width = xmax - xmin
        height = ymax - ymin
        ClassName= row.box_type
        # assign different color to different classes of objects
        if row.box_type == 'False':
            ax.annotate('False', xy=(xmax-40,ymin+20))
            rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = 'r', facecolor = 'none')
        elif row.box_type == 'True':
            ax.annotate('True', xy=(xmax-40,ymin+20))
            rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = 'b', facecolor = 'none')      
        else:
            print("nothing")
    
        ax.add_patch(rect)   
        if not os.path.exists("imagesBox"):
            os.makedirs("imagesBox")

        fig.savefig('imagesBox/' + os.path.splitext(imageFileName)[0] + ".jpg", dpi=90, bbox_inches='tight')
    plt.close()
    print("ImageName: " + imageFileName + " is saved in imagesBox folder")
        
print("PLOTBOX COMPLETED!")
        