import json
import math
import os.path
from typing import List, Dict

import numpy as np
import cv2
import yaml
from yaml.loader import SafeLoader
import argparse
import os
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--class_file", help="YAML file path containing classes", required=True)
parser.add_argument("--output", help="Output YOLO directory", required=True)
parser.add_argument("--input", help="Input COCO directory", required=True)
args = parser.parse_args()


def create_directories(output_folder: str) -> None:
    """
    Create parent directory and its children : images and labelTxt
    :param output_folder: parent directory specified by the user in cmd line
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, "labelTxt")):
        os.makedirs(os.path.join(output_folder, "labelTxt"))
    if not os.path.exists(os.path.join(output_folder, "images")):
        os.makedirs(os.path.join(output_folder, "images"))


def get_classes(class_filename: str) -> Dict:
    """
    Get classes entered by the user reading a YAML file
    :param class_filename:
    :return: dictonnary containing id and respective classes
    """
    with open(class_filename) as f:
        data = yaml.load(f, Loader=SafeLoader)
        return data


def flatten(l: List):
    """
    Flat a 2 dimension list into a 1D list
    :param l: 2D list with 1D list and multiple tuples --> [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    :return: 1D list --> [x0, y0, x1, y1, x2, y2, x3, y3]
    """
    return [item for sublist in l for item in sublist]


if __name__ == '__main__':

    categories = get_classes(args.class_file)
    create_directories(args.output)

    path_ann_folder = os.path.join(args.input, "ann")
    json_filename = os.listdir(path_ann_folder)[0]
    ann_path = os.path.join(path_ann_folder, json_filename)
    with open(ann_path) as jsonFile:
        coco = json.load(jsonFile)
        jsonFile.close()

    for img in tqdm(coco['images']):
        # copy picture
        img_path = os.path.join(args.input, "img", img["file_name"])
        shutil.copyfile(src=img_path, dst=os.path.join(args.output, "images", img["file_name"]))

        idx = img["id"]
        annotations = [v for v in coco['annotations'] if v['image_id'] == idx]

        text = []
        for i in annotations:
            # create annotation file

            points = np.array(i['segmentation'][0], dtype=np.int32)
            # print(points)
            ((cx, cy), (w, h), a) = cv2.minAreaRect(np.array(list(zip(points[0::2], points[1::2]))))

            rotated_box = ((cx, cy), (w, h), a)
            (cnt_x, cnt_y), (w, h), angle = rotated_box

            area = w * h
            theta = angle * math.pi / 180

            theta = -theta
            c = math.cos(theta)
            s = math.sin(theta)
            rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
            # x: left->right ; y: top->down
            rotated_rect = [(s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect]
            yolo_obb_rec = [rotated_rect[2], rotated_rect[3], rotated_rect[0], rotated_rect[1]]
            flat_list = flatten(yolo_obb_rec)
            flat_str_list = [str(nb) for nb in flat_list]

            carac_chain = " ".join(flat_str_list)

            category_number = i['category_id'] - 1
            carac_chain += " " + categories[category_number] + " " + str(category_number)
            text.append(carac_chain)

        with open(os.path.join(args.output, "labelTxt", img["file_name"] + ".txt"), "w") as f:
            for line in text:
                f.write(f"{line}\n")
