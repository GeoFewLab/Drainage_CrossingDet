from pylabel import importer

dataset = importer.ImportVOC(path='./xml')
#dataset.path_to_annotations = './yolov5_labels'
dataset.splitter.GroupShuffleSplit(train_pct=0.8,val_pct=.2)
dataset.export.ExportToYoloV5(output_path='./labels')

