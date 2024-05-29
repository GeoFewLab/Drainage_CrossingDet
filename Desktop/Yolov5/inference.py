import numpy as np
import cv2
import torch
import glob as glob
#from model import create_model
from config import NUM_EPOCHS, SAVE_MODEL_EPOCH, NUM_CLASSES, CLASSES, INFER_FALSE_LABELS,DETECTION_THRESHOLD
import os
import pandas as pd
import copy

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
    
    #NOTE: AFTER TRAINING BASED ON THIS OFFICIAL YOLO TUTORIAL: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#13-organize-directories
    #Load weights from the path which you stored the model weights
    model = torch.hub.load('ultralytics/yolov5', 'custom', './runs/train/yolo_5s9/weights/best.pt')

    model.eval()


    # directory where all the images are present
    DIR_TEST = './IL/pro' #'../validation_data'
    OUT_DIR = './IL/classified_images'  #'../validation_results'
    test_images = glob.glob(f"{DIR_TEST}/*")
    print(f"Test instances: {len(test_images)}")

    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = DETECTION_THRESHOLD

    # create datatable to store output results
    col_names = ['file_name', 'labels','centroids','x1','x2','y1','y2']
    validation_results = pd.DataFrame(columns=col_names)

    for i in range(len(test_images)):
        # get the image file name for saving output later on
        image_name = test_images[i].split('/')[-1].split('.')[0]

        #create new dataframe row
        #validation_results.loc[len(validation_results.index)] = [[],[],[],[],[],[],[]]
        row_dict = {'file_name': None, 'labels':None, 'centroids':[], 'x1':[], 'x2':[], 'y1':[], 'y2':[]}
        #validation_results = validation_results.append(row_dict, ignore_index=True)
        
        #Store current image name
        row_dict['file_name'] = image_name
        print("LOADING:", test_images[i])
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        #image /= 255.0
        # bring color channels to front
        #image = np.transpose(image, (2, 0, 1)).astype(float)
        # convert to tensor
        #image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        #image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
            outputs = outputs.pandas().xyxy[0]
            print(outputs)

        out_dict = {'boxes': [],
                    'scores': [],
                    'labels': []}

        if not outputs.empty:
            cur_box = []
            for i in range(outputs.shape[0]):
                cur_box.append(outputs.iloc[i]['xmin'])
                cur_box.append(outputs.iloc[i]['ymin'])
                cur_box.append(outputs.iloc[i]['xmax'])
                cur_box.append(outputs.iloc[i]['ymax'])
                #cur_box = np.array(cur_box)
                out_dict['boxes'].append(cur_box)
                out_dict['scores'].append(outputs.iloc[i]['confidence'])
                out_dict['labels'].append(outputs.iloc[i]['class'])

                cur_box = []

            out_dict['scores'] = torch.FloatTensor(out_dict['scores'])
            out_dict['boxes'] = torch.FloatTensor(out_dict['boxes'])
            out_dict['labels'] = torch.FloatTensor(out_dict['labels'])
            
        outputs = out_dict
        # load all detection to CPU for further operations
        #outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        #outputs = outputs[0]

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
            #pred_classes = outputs['labels']
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                print(pred_classes[j])
                if(pred_classes[j] == 'False' and INFER_FALSE_LABELS == False):
                    print("Skipping False Label!")
                    continue
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 16)
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
            print(os.path.join(os.getcwd(), str(os.path.basename(OUT_DIR)) , os.path.split(image_name)[1]+'.jpg'))
            print('Save complete: ',cv2.imwrite(os.path.join(os.getcwd(), OUT_DIR.lstrip('./') , os.path.split(image_name)[1]+'.jpg'), orig_image, ))
    
        row_dict['x1'] = x1_list
        row_dict['y1'] = y1_list
        row_dict['x2'] = x2_list
        row_dict['y2'] = y2_list
        row_dict['centroids'] = centroids_list
        row_dict['labels'] = labels_list
        # print("Before append:")
        # print(row_dict)
    
        validation_results = validation_results._append(row_dict, ignore_index=True)
        # print("After append:", type(validation_results))

     
        # validation_results = validation_results.append(row_dict, ignore_index=True)

        print(f"Image {i + 1} done...")
        print('-' * 50)



    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    if not os.path.exists('./IL/'):
        os.makedirs('./IL/')
    print(validation_results)
    validation_results.to_csv('./IL/IL_results.csv')
    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()




