import cv2
import os
from tqdm import tqdm
import json
import numpy as np
from argparse import ArgumentParser

ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]

def parse_inputs():
    """ Parser function to take care of the inputs """
    parser = ArgumentParser(description='Argument: python transform_images.py <data_direction> <output_dir>')
    parser.add_argument('data_dir', type=str,
                        help='Enter path to data direction.')
    parser.add_argument('output_dir', type=str,
                        help='Enter the path of the output of transformation.')
    args = parser.parse_args()

    return (args.data_dir, args.output_dir)

def transform_images():
    
    input_dir, output_dir = parse_inputs()
    pixel_border = 40

    if (not os.path.isdir(output_dir)):
        os.mkdir(output_dir)

    img_list = []

    for extension in ALLOWED_EXTENSIONS:
        img_list.extend([f for f in os.listdir(input_dir) if extension in f])

    for f in tqdm(img_list):

        img = cv2.imdecode(np.fromfile(os.path.join(input_dir, f), dtype=np.uint8), cv2.IMREAD_COLOR)

        dst = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        dst= cv2.copyMakeBorder(dst, pixel_border, pixel_border, pixel_border, pixel_border, cv2.BORDER_CONSTANT,value=(255,255,255))

        for extension in ALLOWED_EXTENSIONS:
            if (extension in f):
                fi = open(os.path.join(input_dir, f).replace(extension, ".json"), "r", encoding = "utf-8")

        data = json.load(fi)

        data["imageHeight"] += pixel_border * 2
        data["imageWidth"] += pixel_border * 2

        annotations = data["shapes"]
        for i in range(len(annotations)):
            point = annotations[i]["points"][0]
            point[0] += pixel_border
            point[1] += pixel_border
            data["shapes"][i]["points"] = [point]

        for extension in ALLOWED_EXTENSIONS:
            if (extension in f):
                fo = open(os.path.join(output_dir, f).replace(extension, ".json"), "w", encoding = "utf-8")
        
        json.dump(data, fo, indent = 4)
        fo.close()
        
        cv2.imwrite(os.path.join(output_dir, f), dst)


if __name__ == "__main__":
    transform_images()