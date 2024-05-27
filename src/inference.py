import numpy as np
import cv2
import torch
import glob as glob
from model import create_model
from config import NUM_EPOCHS, SAVE_MODEL_EPOCH, NUM_CLASSES, CLASSES, INFER_FALSE_LABELS
import os
import pandas as pd

'''
NOTE: Relative paths are used in this file. These paths are relative to the root directory. If you intend on running
        This code from this file specifically, you must change them to go back a directory 
        (ie. ../outputs/model rather than ./outputs/model)
         '''

def inference():
    #compute the latest saved model based on the number of epochs and saving interval
    #np.floor((NUM_EPOCHS/SAVE_MODEL_EPOCH)*SAVE_MODEL_EPOCH)
    # set the computation device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(
        './outputs/model'+str(NUM_EPOCHS)+'.pth', map_location=device
    ))
    model.eval()


    # directory where all the images are present
    DIR_TEST = './big_geo_data/val' #'../validation_data'
    OUT_DIR = './big_geo_data/classified_images'  #'../validation_results'
    test_images = glob.glob(f"{DIR_TEST}/*")
    print(f"Test instances: {len(test_images)}")

    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = 0.8

    # create datatable to store output results
    col_names = ['file_name', 'labels','centroids','x1','x2','y1','y2']
    validation_results = pd.DataFrame(columns=col_names)

    for i in range(len(test_images)):
        # get the image file name for saving output later on
        image_name = test_images[i].split('/')[-1].split('.')[0]

        #create new dataframe row
        validation_results.loc[validation_results.shape[0]] = [None,None,None,None,None,None,None]
        #Store current image name
        validation_results['file_name'][i] = image_name
        print("LOADING:", test_images[i])
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(float)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        outputs = outputs[0]

        print(outputs)


        #holds on to all info for all boxes in an image
        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []
        centroids_list = []
        labels_list = []
        # carry further only if there are detected boxes
        if len(outputs['boxes']) != 0:
            boxes = outputs['boxes'].data.numpy()
            scores = outputs['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs['labels'].cpu().numpy().astype(int)]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                print(pred_classes[j])
                if(pred_classes[j] == 'False' and INFER_FALSE_LABELS == False):
                    print("Skipping False Label!")
                    continue
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j],
                            (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2, lineType=cv2.LINE_AA)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)
                labels_list.append(pred_classes[j])
                #centroid of a box should be the average of its 2 corners (an x,y tuple)
                centroids_list.append( ((x1+x2)/2, (y1+y2)/2) )

            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(1)
            print(os.path.join(os.getcwd(),os.pardir, str(os.path.basename(OUT_DIR)) , os.path.split(image_name)[1]+'.jpg'))
            print('Save complete: ',cv2.imwrite(os.path.join(os.getcwd(),os.pardir, OUT_DIR.lstrip('./') , os.path.split(image_name)[1]+'.jpg'), orig_image, ))
    
        validation_results['x1'][i] = x1_list
        validation_results['y1'][i] = y1_list
        validation_results['x2'][i] = x2_list
        validation_results['y2'][i] = y2_list
        validation_results['centroids'][i] = centroids_list
        validation_results['labels'][i] = labels_list

        print(f"Image {i + 1} done...")
        print('-' * 50)


    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    validation_results.to_csv('./validation_results/validation_results.csv')
    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()




