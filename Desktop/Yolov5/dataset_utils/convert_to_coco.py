import os
import shutil
import xml.etree.ElementTree as ET

from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json


def create_annotation_json_file(coco, source_path, destination_path, width, height, xmin, xmax, category_id, label):
    coco_image = CocoImage(file_name=source_path, height=width, width=height)

    coco_image.add_annotation(
        CocoAnnotation(
            bbox=[0, 0, width, height],
            category_id=category_id,
            category_name=label,
        )
    )

    coco.add_image(coco_image)
    save_json(data=coco.json, save_path=destination_path)


def get_xml_data(path):
    tree = ET.parse(path)
    root = tree.getroot()

    # Extracting width and height
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Extracting object information
    objects = root.findall("object")
    annotations = []
    for obj in objects:
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        annotations.append({
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

    return width, height, annotations


def main():
    coco = Coco()

    coco.add_category(CocoCategory(id=0, name="True"))
    coco.add_category(CocoCategory(id=1, name="Background"))

    ROOT_DATA_DIR = './xml'#os.path.join("")
    ANNOTATION_PATH = './coco_labels'#os.path.join("", "xml")
    TRAIN_PATH = os.path.join("./data", "train")
    VAL_PATH = os.path.join("./data", "val")

    os.makedirs(ANNOTATION_PATH, exist_ok=True)
    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(VAL_PATH, exist_ok=True)

    for file_name in os.listdir(ROOT_DATA_DIR):
        i = int(file_name.split(".")[0])

        source_path = os.path.join(ROOT_DATA_DIR, file_name)
        train_destination_path = os.path.join(TRAIN_PATH, file_name)
        val_destination_path = os.path.join(VAL_PATH, file_name)
        annotation_destination_path = os.path.join(ANNOTATION_PATH, f"{i}.json")

        if file_name.endswith(".tif"):
            if i <= 1720:
                shutil.copy(source_path, train_destination_path)
            else:
                shutil.copy(source_path, val_destination_path)
        elif file_name.endswith(".xml"):
            width, height, annotations = get_xml_data(source_path)

            label = annotations[0]["name"]
            xmin = annotations[0]["xmin"]
            ymin = annotations[0]["ymin"]
            xmax = annotations[0]["xmax"]
            ymax = annotations[0]["ymax"]

            category_id = 0
            if label == "True":
                category_id = 0
            elif label == "Background":
                category_id = 1

            create_annotation_json_file(coco, source_path, annotation_destination_path, width, height, xmin, xmax, category_id, label)


if __name__ == "__main__":
    main()