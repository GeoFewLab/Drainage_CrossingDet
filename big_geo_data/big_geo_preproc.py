

import numpy as np
import pandas as pd
import os 
from pascal_voc_writer import Writer


def big_geo_preproc():
    out_dir = './annotations'
    df = pd.read_csv('./coordinates_Bbox.csv')

    # Convert bottom left origin to top left origin, by subtracting 800 from all y coordinates
    # (this simply translates the origin 800 up on the y axis, the height of our images)
    BOUND_SIZE = 50
    image_height = 800
    image_width = 800
    #image_height_translation = 0
    df['Culvert Local Y'] = 800 - df['Culvert Local Y']

    df = df.groupby(['Sample ID','Sample Name']).aggregate(lambda x: list(x)).reset_index()


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #generate xml files in pascal voc format
    def generate_xml(path,id,x,y,n):
        #generate 2n-by-2n bounding boxes around center coordinates and save to xml
        writer = Writer(path,image_width,image_height)
        for i in range(len(x)):
            #using np.clip to bound the values inside image dimensions
            xmin = int(np.clip(x[i] - n,0,image_width))
            ymin = int(np.clip(y[i] - n,0,image_height))
            xmax = int(np.clip(x[i] + n,0,image_width))
            ymax = int(np.clip(y[i] + n,0,image_height))

            writer.addObject('True',xmin,ymin,xmax,ymax)
            #print('saving to: '+out_dir+'/'+str(id)+'.xml')
            writer.save(out_dir+'/'+str(id)+'.xml')
            

    df.apply(lambda x: generate_xml('./Sample800_norm/'+x['Sample Name'],os.path.splitext(x['Sample Name'])[0],x['Culvert Local X'],x['Culvert Local Y'],BOUND_SIZE),axis=1)



if __name__ == '__main__':
    big_geo_preproc()

