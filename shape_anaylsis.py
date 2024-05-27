import os
from turtle import left, right
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import torch
import numpy as np
import cv2
import os
from PIL import Image
from src.config import NUM_EPOCHS, NUM_CLASSES, IMAGE_SPLIT_DIM

from src.model import create_model
#import earthpy as et
from shapely.geometry import Point, Polygon, box



#Normalize image pixel values so that we can see them
def disparity_normalization(disp): # disp is an array in uint8 data type
        # disp_norm = cv2.normalize(src=disp, dst= disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        _min = np.amin(disp)
        _max = np.amax(disp)
        #disp_norm = disp - _min * 255.0 / (_max - _min)
        disp_norm = (disp - _min) * 255.0 / (_max - _min)
        disp_norm = np.uint8(disp_norm)
        #plt.imshow(disp_norm)
        #plt.show()

        return disp_norm 


im = Image.open('./small_geo_data/mosaic_dem800.tif')
# Normalise to range 0..255
#norm = (im.astype(np.float)-im.min())*255.0 / (im.max()-im.min())
norm = disparity_normalization(im)


if not os.path.exists('./small_geo_data/normalized_images'):
    os.makedirs('./small_geo_data/normalized_images')

converted_image = Image.fromarray(norm)
converted_image.save("./small_geo_data/normalized_images/sample1.jpg")


#################  Generate smaller sub-images for individual classification #####################


if not os.path.exists('./small_geo_data/split_images'):
    os.makedirs('./small_geo_data/split_images')

# TODO replace this to make a seperate directory for each split up image
if not os.path.exists('./small_geo_data/split_images/sample1'):
    os.makedirs('./small_geo_data/split_images/sample1')

img = cv2.imread('./small_geo_data/normalized_images/sample1.jpg')
for r in range(0,img.shape[0],IMAGE_SPLIT_DIM):
    for c in range(0,img.shape[1],IMAGE_SPLIT_DIM):
        cv2.imwrite(f"./small_geo_data/split_images/sample1/img{r}_{c}.png",img[r:r+IMAGE_SPLIT_DIM, c:c+IMAGE_SPLIT_DIM,:])

#################  Create validation image #####################

points = pd.read_excel('./small_geo_data/point.xlsx')

#points = pd.read_csv('./small_geo_data/point.csv')
print(points.iloc[2])

#Extract mapping bounds
top_bound = points.iloc[0]['Unnamed: 1'] 
bottom_bound = points.iloc[1]['Unnamed: 1']
left_bound = points.iloc[2]['Unnamed: 1']
right_bound = points.iloc[3]['Unnamed: 1']
height = top_bound - bottom_bound
#Flip the order (index 3 minus index 2) because axes increases from left to right (we want right - left for a positive value)
width =  right_bound - left_bound
print("HEIGHT: ",height)
print('WIDTH: ', width)

# Extract list of coordinate points (loop through tbbox central points and fbox central points)
true_points = []
false_points = []

# Note: "No is just short for number in the specific excel sheet this is for"
for i in range(points['X'].shape[0]):
    if(np.isnan(points.iloc[i]['X']) == False):
        coord_true = Point(points.iloc[i]['X'],points.iloc[i]['Y'])
        true_points.append(coord_true)
    if(np.isnan(points.iloc[i]['X.1']) == False):
        coord_false = Point(points.iloc[i]['X.1'],points.iloc[i]['Y.1'])
        false_points.append(coord_false)

print("Number of TRUE points: ", len(true_points))

print("Number of FALSE points:  ", len(false_points))

#Use bounds to generate bounding box for image

geodata_true = gpd.GeoDataFrame(true_points,geometry=true_points)
geodata_false = gpd.GeoDataFrame(false_points,geometry=false_points)


#fig, ax = plt.subplots()
img = plt.imread('./small_geo_data/normalized_images/sample1.jpg')
ax = plt.axes()
xlim = ([left_bound,right_bound])
ylim = ([bottom_bound,top_bound])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.imshow(img, extent=[left_bound,right_bound,bottom_bound,top_bound])
geodata_true.plot(ax = ax, color='yellow')
geodata_false.plot(ax = ax, color='red')
#plt.show()


if not os.path.exists('./small_geo_data/labeled_images'):
    os.makedirs('./small_geo_data/labeled_images')

plt.savefig('./small_geo_data/labeled_images/sample1.jpg')

