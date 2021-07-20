#! -*- coding:utf-8 -*-

import numpy as np
import re
import os
from tqdm import tqdm
import random
import json
from argparse import ArgumentParser
import math


ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]

def parse_inputs():
    """ Parser function to take care of the inputs """
    parser = ArgumentParser(description='Argument: python data_preprocess.py <data_direction> <output_annotation_path> <test_ratio>')
    parser.add_argument('data_dir', type=str,
                        help='Enter path to data direction.')
    parser.add_argument('output_annotation_path', type=str,
                        help='Enter the path of the output of annotation files.')
    parser.add_argument('test_ratio', default=0.1, type=float,
                        help='Test Ratio.')
    args = parser.parse_args()

    return (args.data_dir, args.output_annotation_path, args.test_ratio)


def distance_two_points(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def get_center_point(points):
    center_point = [0, 0]
    for point in points:
        center_point[0] += point[0] / len(points)
        center_point[1] += point[1] / len(points)
    return np.array(center_point)


def shrinking_points(points, change_pixel):

    center_point = get_center_point(points)
    distance_from_corner_to_center = distance_two_points(points[0], center_point)

    increase_ratio = (change_pixel + distance_from_corner_to_center) / distance_from_corner_to_center
    new_points = []
    for point in points:
        new_point = (np.array(point) - center_point) * increase_ratio + center_point
        new_points.append(new_point.tolist())

    return new_points



def train_test_split(image_dir, test_ratio, val_ratio = 0.2):
    folder_dirs = [f for f in os.listdir(image_dir)]
    images = []
    for folder in folder_dirs:
        images.extend([os.path.join(folder + "/", f) for f in os.listdir(os.path.join(image_dir, folder))
                    if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)])

    random.seed(1234)
    random.shuffle(images)

    batch_size = 32
    
    train_num = int(len(images) * (1 - val_ratio - test_ratio) // batch_size * batch_size)
    val_num = int(len(images) * val_ratio)
    images_train = images[:train_num]
    images_val = images[train_num:train_num + val_num]
    images_test = images[train_num + val_num:]

    return images_train, images_val, images_test


def parse_annotation(data_dir, image_list, output_annotation):

    folder_dirs = [f for f in os.listdir(data_dir)]
    json_file = []
    for tail_img in ALLOWED_EXTENSIONS:
        json_file.extend([f.replace(tail_img, ".json") for f in image_list if tail_img in f])
    
    result_str = []
    print("Getting Annotations {}...".format(output_annotation))
    for f in tqdm(json_file):
        str_data = []
        fi = open(os.path.join(data_dir, f), "r", encoding = "utf-8")
        data = json.load(fi)
        

        str_data.append(os.path.join(data_dir, f.replace(".json", ".jpg")))
        annotations = data["shapes"]
        width = data["imageWidth"]
        height = data["imageHeight"]


        points = []
        for i in range(len(annotations)):
            points.append(annotations[i]["points"][0])

        center_point = get_center_point(points)

        thresh = distance_two_points(center_point, points[0]) / 8
        shrinking_thresh = thresh * 5 / 4

        points = shrinking_points(points, -shrinking_thresh)

        for i in range(len(annotations)):
            label = annotations[i]["label"]
            if (label not in ["top_left", "top_right", "bottom_left", "bottom_right"]):
                continue

            point = points[i]
            x1 = int(max(point[0] - thresh, 0))
            x2 = int(min(point[0] + thresh, width - 1))
            y1 = int(max(point[1] - thresh, 0))
            y2 = int(min(point[1] + thresh, height - 1))

            str_data.extend([label, str(x1), str(y1), str(x2), str(y2)])


        str_data = ",".join(str_data)
        result_str.append(str_data)


    result_str = "\n".join(result_str)
    
    fo = open(output_annotation, "w", encoding = "utf-8")
    fo.write(result_str)
    fo.close()

if __name__ == "__main__":

    #!python convert_pascalvoc_dataset/data_preprocess.py "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/cmnd_back/" "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/Annotations/" 0.1

    # OR test on local:
    # python convert_pascalvoc_dataset/data_preprocess.py "../cmnd_back/" "../Annotations/" 0.1
    data_dir, output_annotation_path, test_ratio = parse_inputs()

    if (not os.path.isdir(output_annotation_path)):
        os.mkdir(output_annotation_path)

    train_annotaion_file = os.path.join(output_annotation_path, "Train_annotation.txt")
    val_annotation_file = os.path.join(output_annotation_path, "Val_annotation.txt")
    test_annotation_file = os.path.join(output_annotation_path, "Test_annotation.txt")

    train_img, val_img, test_img = train_test_split(data_dir, test_ratio)
    parse_annotation(data_dir, train_img, train_annotaion_file)
    parse_annotation(data_dir, val_img, val_annotation_file)
    parse_annotation(data_dir, test_img, test_annotation_file)