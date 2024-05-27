import os

import numpy as np
import os
import tifffile as tiff
from PIL import Image
import imageio.v2 as imageio
from tqdm import tqdm
import cv2

def normalize_images():
    PATH = './Sample800'
    OUT_PATH = './Sample800_norm'
    FILE_TYPE = '.tif'

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)


    def apply_sobel(image,RGB=False):
        #Apply sobel filter to refine edges
        if(RGB == False):
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return grad

    def normalize_image(img_path,out_path):

        image = tiff.imread(img_path)      

        
         # Calculate the histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=(image.min(), image.max()))

        # Calculate the Cumulative Distribution Function (CDF)
        cdf = hist.cumsum()

        # Normalize the CDF to have values between 0 and 255
        cdf_normalized = cdf * 255 / cdf[-1]

        # Use the CDF values to map the pixel values
        equ_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)

        # Reshape the image back to its original shape
        equalized_image = equ_image.reshape(image.shape)

        image = equalized_image.astype(np.uint8)


        #image = apply_sobel(image)

        # convert back to PIL for saving
        image = Image.fromarray(image)
        image.save(out_path, "TIFF")
        return image


    file_list = os.listdir(PATH)

    for filename in tqdm(file_list):
        if filename.endswith(FILE_TYPE):
            # Normalise to range 0..255
            norm = normalize_image(PATH+'/'+filename,OUT_PATH+'/'+filename)

       
if __name__ == '__main__':
    normalize_images()