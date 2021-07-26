import glob
import os
import time

import torch
import cv2
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset, MyDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
import collections


def distance_two_points(point_1, point_2):
    return np.sqrt(np.power(point_1[0] - point_2[0], 2) + np.power(point_1[1] - point_2[1], 2))

def get_center_bbox(box):
    a = (box[0] + box[2]) / 2
    b = (box[1] + box[3]) / 2
    return np.array([a, b])


def check_point(point, image):
    w = image.shape[1]
    h = image.shape[0]

    if (point[0] < 0):
        point[0] = 0
    elif (point[0] > w):
        point[0] = w - 1
    
    if (point[1] < 0):
        point[1] = 0
    elif (point[1] > h):
        point[1] = h - 1

    return point

def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))
    cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return dst


def align_image(image, top_left, top_right, bottom_right, bottom_left, expand_alignment = False):
    top_left_point = get_center_bbox(top_left)
    top_right_point = get_center_bbox(top_right)
    bottom_right_point = get_center_bbox(bottom_right)
    bottom_left_point = get_center_bbox(bottom_left)

    if (expand_alignment):
        x_val = (top_left_point[0] + top_right_point[0] + bottom_right_point[0] + bottom_left_point[0]) / 4
        y_val = (top_left_point[1] + top_right_point[1] + bottom_right_point[1] + bottom_left_point[1]) / 4
        center_point = np.array([x_val, y_val])

        distance_from_corner_to_center = distance_two_points(top_left_point, center_point)
        increase_pixel = distance_from_corner_to_center / 4.5
        increase_ratio = (increase_pixel + distance_from_corner_to_center) / distance_from_corner_to_center

        top_left_point = (top_left_point - center_point) * increase_ratio + center_point
        top_right_point = (top_right_point - center_point) * increase_ratio + center_point
        bottom_right_point = (bottom_right_point - center_point) * increase_ratio + center_point
        bottom_left_point = (bottom_left_point - center_point) * increase_ratio + center_point

    top_left_point = check_point(top_left_point, image)
    top_right_point = check_point(top_right_point, image)
    bottom_right_point = check_point(bottom_right_point, image)
    bottom_left_point = check_point(bottom_left_point, image)

    source_points = np.float32(
        [top_left_point, top_right_point, bottom_right_point, bottom_left_point]
    )
    crop = perspective_transform(image, source_points)
    return crop


def image_processing(image):

    # # Detail enhance and create border
    # dst = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    # dst= cv2.copyMakeBorder(dst, pixel_border, pixel_border, pixel_border, pixel_border, cv2.BORDER_CONSTANT,value=(255,255,255))

    dst = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dst = Image.fromarray(dst)
    return np.asarray(dst)

def process_duplicate_labels(labels, scores, boxes):
    list_duplicate = [item for item, count in collections.Counter(labels).items() if count > 1]
        
    for dup in list_duplicate:
        list_indices = [i for (i, item) in enumerate(labels) if item == dup]
        max_conf_indice = list_indices[0]
        for indice in list_indices:
            if scores[indice] > scores[max_conf_indice]:
                max_conf_indice = indice
                
        list_indices.remove(max_conf_indice)
        for index in sorted(list_indices, reverse=True):
            labels = np.delete(labels, index)
            scores = np.delete(scores, index)
            boxes = np.delete(boxes, index, 0)

    return labels, scores, boxes

@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type, check_9_labels=False):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    elif dataset_type == "custom":
        if (check_9_labels):
            class_names = MyDataset.class_names_9_labels
        else:
            class_names = MyDataset.class_names_5_labels
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    folder_path = [f for f in os.listdir(images_dir)]

    image_paths = []
    for folder in folder_path:

        image_paths.extend(glob.glob(os.path.join(images_dir, folder, '*.jpg')))

        result_output_dir = os.path.join(output_dir, "result/", folder)
        mkdir(result_output_dir)
        output_dir_crop = os.path.join(output_dir, 'crop/', folder)
        mkdir(output_dir_crop)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()


    count_true = 0
    count_error_1 = 0
    count_error_more_2 = 0

    error_images = []
    images_missing_1_corner = []

    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # image_show = image.copy()
        # cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
        
        # width = image.shape[1]
        # height = image.shape[0]
        # ratio_resize = 1
        # if (width * height > 6 * 10**6):
        #     ratio_resize = 4
        # elif (width * height > 8 * 10**5):
        #     ratio_resize = 1.5
        
        # image = cv2.resize(image, (int(width / ratio_resize), int(height / ratio_resize)))


        dst = image_processing(image)

        height, width = dst.shape[:2]
        images = transforms(dst)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))
        
        labels, scores, boxes = process_duplicate_labels(labels, scores, boxes)

        # for i in range(len(boxes)):
        #     for k in range(len(boxes[i])):
        #         boxes[i][k] -= pixel_border
        #         boxes[i][k] *= ratio_resize
        

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "result", os.path.basename(os.path.dirname(image_path)), image_name), drawn_image)
        # Crop image
        # image = image_show.copy()
        pair = zip(labels, boxes)
        sort_pair = sorted(pair)
        boxes = [element for _, element in sort_pair]
        labels = [element for element, _ in sort_pair]
        
        crop = None
        if len(boxes) == 4:
            count_true += 1
            crop = align_image(image, boxes[0], boxes[1], boxes[2], boxes[3], True)
        elif len(boxes) == 3:
            # Find fourth missed corner
            thresh = 0
            images_missing_1_corner.append(os.path.join(os.path.basename(os.path.dirname(image_path)), image_name))
            count_error_1 += 1
            
            if 1 not in labels:
                midpoint = np.add(get_center_bbox(boxes[0]), get_center_bbox(boxes[2])) / 2
                y = int(2 * midpoint[1] - get_center_bbox(boxes[1])[1] + thresh)
                x = int(2 * midpoint[0] - get_center_bbox(boxes[1])[0] + thresh)
                TL = np.array([x, y, x, y])
                crop = align_image(image, TL, boxes[0], boxes[1], boxes[2], True)
            elif 2 not in labels:
                midpoint = np.add(get_center_bbox(boxes[0]), get_center_bbox(boxes[1])) / 2
                y = int(2 * midpoint[1] - get_center_bbox(boxes[2])[1] + thresh)
                x = int(2 * midpoint[0] - get_center_bbox(boxes[2])[0] + thresh)
                TR = np.array([x, y, x, y])
                crop = align_image(image, boxes[0], TR, boxes[1], boxes[2], True)
            elif 3 not in labels:
                midpoint = np.add(get_center_bbox(boxes[2]), get_center_bbox(boxes[1])) / 2
                y = int(2 * midpoint[1] - get_center_bbox(boxes[0])[1] + thresh)
                x = int(2 * midpoint[0] - get_center_bbox(boxes[0])[0] + thresh)
                BR = np.array([x, y, x, y])
                crop = align_image(image, boxes[0], boxes[1], BR, boxes[2], True)
            elif 4 not in labels:
                midpoint = np.add(get_center_bbox(boxes[0]), get_center_bbox(boxes[2])) / 2
                y = int(2 * midpoint[1] - get_center_bbox(boxes[1])[1] + thresh)
                x = int(2 * midpoint[0] - get_center_bbox(boxes[1])[0] + thresh)
                BL = np.array([x, y, x, y])
                crop = align_image(image, boxes[0], boxes[1], boxes[2], BL, True)
        else:
            count_error_more_2 += 1
            error_images.append(os.path.join(os.path.basename(os.path.dirname(image_path)), image_name))
            print("Please take a photo again, number of detected corners is:", len(boxes))
            continue

        folder = os.path.basename(os.path.dirname(image_path))
        cv2.imwrite(os.path.join(output_dir, "crop", folder, image_name), crop)

    print("Number of true images: {}".format(count_true))
    print("Number of 3 corner images: {}".format(count_error_1))
    print("Number of 2 corner images: {}".format(count_error_more_2))

    print("Image have 3 corners: {}".format(images_missing_1_corner))
    print("Error Images: {}".format(error_images))


def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result/', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="custom", type=str, help='Specify dataset type. Currently support voc and coco.')
    parser.add_argument("--check_9_labels", default=False, action="store_true", help='Allow the dataset of 9 labels (4 corners of 2 face including top and back of id card)')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type,
             check_9_labels=args.check_9_labels)


if __name__ == '__main__':
    main()
