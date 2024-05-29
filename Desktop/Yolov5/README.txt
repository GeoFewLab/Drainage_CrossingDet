
This branch is used to run analysis on YOLOv5 models. Please refer to the YOLOv5 documentation and
tutorial on training the model. One such tutorial can be found on the YOLOv5 github repository here:

https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

NOTE 1: You will likely have to manipulate the data format from its original form compared to what was
used for faster-rcnn. There are some scripts in the ./dataset_utils directory
to help you with this.

NOTE 2: To make the scripts more helpful, start by generating the faster-rcnn data in the main branch,
then use the scripts to change them from that format to the desired YOLO format.


1. Place the trained YOLOv5 model weights in the respective 'outputs/model/' directory
2. Ensure all images are correctly labeled and are divided into their typical directories as per instructions in the README, such as they are after running steps on data preparation.
3. Run validate.py to generate metrics.

Should you need any tools to convert training and validation data formats between those used in faster-rcnn and the MS COCO used in YOLOv5, there are a couple scripts in the dataset_utils directory.

