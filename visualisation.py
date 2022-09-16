import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from tqdm import tqdm


def parse_opt() -> argparse.Namespace:
    """
    Argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Path of the input picture or input directory of images", required=True)
    parser.add_argument("--output_path", type=str, help="Output directory of annotated pictures", required=True)
    parser.add_argument("--draw_r", type=bool, help="Draw the oriented bounding box", default=True)
    parser.add_argument("--draw_sa", type=bool, help="Draw the main symmetric axis of bounding box", default=False)
    args = parser.parse_args()
    return args


def get_two_min(lengths: List[int]) -> Tuple[int, int]:
    """
    Get the two minimum lengths index in a list of vector norms
    Ex. input [50, 51, 109, 123]  --> output [0, 1]
    :param lengths: norm of 4 vectors
    :return: index of the two smallest vectors in terms of length
    """
    min_index1 = lengths.index(min(lengths))
    lengths[min_index1] = np.inf
    min_index2 = lengths.index(min(lengths))
    return min_index1, min_index2


def draw_diagonal(len_lines: List[int], rotated_rect: List[float], img: np.ndarray) -> None:
    """
    Draw principal symmetric axis of a bounding boxes
    :param len_lines: norm of each rectangle's side
    :param rotated_rect: list of rectangles corners in format [x1, y1, x2, y2, x3, y3, x4, y4]
    :param img: input picture in OpenCV format
    """
    index_width = get_two_min(len_lines)
    pts = []
    for i in index_width:
        j = (i * 2 + 2) % 8

        mid_x = round((rotated_rect[i * 2] + rotated_rect[j]) / 2)
        mid_y = round((rotated_rect[i * 2 + 1] + rotated_rect[j + 1]) / 2)
        cv2.circle(img, (mid_x, mid_y), radius=0, color=(0, 0, 255), thickness=-1)

        pts.append((mid_x, mid_y))

    cv2.line(img, pts[0], pts[1], color=(0, 0, 255), thickness=2)


def create_picture(path_img: str, output_path: str, draw_r: bool, draw_d: bool) -> None:
    """
    Create annotated picture in function of the labels
    :param path_img: path of input picture
    :param output_path: path of the output directory
    :param draw_r: boolean which indicates if the bounding boxes will be displayed
    :param draw_d: boolean which indicates if the symmetric axis will be displayed
    """
    img = cv2.imread(path_img)

    root = os.path.dirname(os.path.dirname(path_img))
    filename = path_img.split("\\")[-1]
    path_txt = os.path.join(root, "labelTxt", filename[:-4] + ".txt")

    with open(path_txt) as f:
        lines = f.readlines()

    rectangles = []
    for l in lines:
        rectangles.append(np.array(l.split(" ")[:-2], dtype=float))

    for rotated_rect in rectangles:
        len_lines = []
        for k in range(0, 8, 2):
            j = (k + 2) % 8
            if draw_r:
                img = cv2.line(img, (round(rotated_rect[k]), round(rotated_rect[k + 1])),
                               (round(rotated_rect[j]), round(rotated_rect[j + 1])), (255, 0, 0), 1)

            if draw_d:
                diff_x = (rotated_rect[k] - rotated_rect[j]) ** 2
                diff_y = (rotated_rect[k + 1] - rotated_rect[j + 1]) ** 2
                len_lines.append(np.sqrt(diff_x + diff_y))

        if draw_d:
            draw_diagonal(len_lines, rotated_rect, img)

    output_file = os.path.join(output_path, filename)
    cv2.imwrite(output_file, img)


def run(source, output_path, draw_r, draw_sa):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if os.path.isdir(source):
        for i in tqdm(os.listdir(source)):
            create_picture(os.path.join(source, i), output_path, draw_r, draw_sa)

    else:
        create_picture(source, output_path, draw_r, draw_sa)


def main(opt: argparse.Namespace):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
