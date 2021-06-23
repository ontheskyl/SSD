#! -*- coding: utf-8 -*-

import os
import sys
from easydict import EasyDict as edict

from convert_pascalvoc_dataset.pascal_voc import PASCALVOC07
from argparse import ArgumentParser

config = edict()

config.author = "sunshine tech"
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
    parser = ArgumentParser(description='Argument: python data_preprocess.py <output_direction> <annotation_path> <validation_ratio>')
    parser.add_argument('output_dir', type=str,
                        help='Enter the path of the output.')
    parser.add_argument('annotation_dir', default="Annotations", type=str,
                        help='Enter the path of annotation files.')
    parser.add_argument('val_ratio', default=0.0, type=float,
                        help='Validation Ratio.')
    args = parser.parse_args()

    return (args.output_dir, args.annotation_dir, args.val_ratio)


if __name__ == "__main__":

    # !python build.py "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/data" "/content/drive/MyDrive/Colab Notebooks/Sunshine Tech/Annotations" 0

    output_dir, annotation_dir, val_ratio = parse_inputs()

    if (not os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    trainval_anno = os.path.join(annotation_dir, 'Train_annotation.txt')
    test_anno = os.path.join(annotation_dir, 'Test_annotation.txt')

    p = PASCALVOC07(trainval_anno, test_anno, val_ratio, output_dir, config)
    p.build(True)
