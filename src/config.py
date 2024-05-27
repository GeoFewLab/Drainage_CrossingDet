

'''Based on the following tutorial https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/'''


import torch
BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 800 #512 # resize the image for training and transforms
NUM_EPOCHS = 10 # number of epochs to train for
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '../big_geo_data/train'
# validation images and XML files directory
VALID_DIR = '../big_geo_data/test'

IMAGE_TYPE = 'tif'
# classes: 0 index is reserved for 'background' used in the pytorch faster_rcnn model
CLASSES = [
    'Background','True'
]
NUM_CLASSES = len(CLASSES)
# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = './outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs


IMAGE_SPLIT_DIM = 400 # nxn dimension to split larger shape files into, for classification
INFER_FALSE_LABELS = False

