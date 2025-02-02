#! -*- coding: utf-8 -*-

import os
import sys
from argparse import ArgumentParser
from easydict import EasyDict as edict

from pascal_voc.pascal_voc import PASCALVOC07

config = edict()

config.author = "Sunshine Tech"
config.root = "annotation"
config.folder = "VOC2007"
config.annotation = "PASCAL VOC2007"
config.segmented = "0"
config.difficult = "0"
config.truncated = "0"
config.pose = "Unspecified"
config.database = "CMND_BACK"
config.depth = "3"


def parse_inputs():

    """ Parser function to take care of the inputs """

    parser = ArgumentParser(description='Argument: python data_preprocess.py <annotation_path> <output_direction>')
    
    parser.add_argument('annotation_dir', default="Annotations", type=str,
                        help='Enter the path of annotation files.')
    parser.add_argument('output_dir', type=str,
                        help='Enter the path of the output.')
    args = parser.parse_args()

    return (args.output_dir, args.annotation_dir)


if __name__ == "__main__":

    # !python convert_pascalvoc_dataset/build.py "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/Annotations" "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/data"

    # OR test on local
    # python convert_pascalvoc_dataset/build.py "Annotations/" "data/"

    output_dir, annotation_dir = parse_inputs()

    if (not os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    print("Building PASCAL VOC 2007...")
    trainval_anno = os.path.join(annotation_dir, 'Train_annotation.txt')
    val_anno = os.path.join(annotation_dir, 'Val_annotation.txt')
    test_anno = os.path.join(annotation_dir, 'Test_annotation.txt')

    p = PASCALVOC07(trainval_anno, val_anno, test_anno, output_dir, config)
    p.build(True)
