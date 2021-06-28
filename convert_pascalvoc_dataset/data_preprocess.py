#! -*- coding:utf-8 -*-

import re
import os
from tqdm import tqdm
import random
import json
from argparse import ArgumentParser
import math

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

def train_test_split(image_dir, test_ratio):
    images = [f for f in os.listdir(image_dir)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    random.seed(1234)
    random.shuffle(images)

    batch_size = 32
    
    ratio = int(len(images) * (1 - test_ratio) // batch_size * batch_size)
    images_train = images[:ratio]
    images_test = images[ratio:]

    return images_train, images_test

def parse_annotation(data_dir, image_list, output_annotation):

    json_file = [f.replace(".jpg", ".json") for f in image_list]
    
    result_str = []
    print("Getting Annotations")
    for f in tqdm(json_file):
        str_data = []
        fi = open(os.path.join(data_dir, f), "r")
        data = json.load(fi)

        str_data.append(os.path.join(data_dir, data["imagePath"]))
        annotations = data["shapes"]
        width = data["imageWidth"]
        height = data["imageHeight"]

        thresh = math.log2(width * height) * 1.8

        for ann in annotations:
            label = ann["label"]
            point = ann["points"][0]
            x1 = int(max(point[0] - thresh, 0))
            x2 = int(min(point[0] + thresh, width - 1))
            y1 = int(max(point[1] - thresh, 0))
            y2 = int(min(point[1] + thresh, height - 1))

            str_data.extend([label, str(x1), str(y1), str(x2), str(y2)])

        str_data = ",".join(str_data)
        result_str.append(str_data)

    result_str = "\n".join(result_str)
    
    fo = open(output_annotation, "w")
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
    test_annotation_file = os.path.join(output_annotation_path, "Test_annotation.txt")

    train_img, test_img = train_test_split(data_dir, test_ratio)
    parse_annotation(data_dir, train_img, train_annotaion_file)
    parse_annotation(data_dir, test_img, test_annotation_file)