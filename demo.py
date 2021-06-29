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


def get_center_bbox(box):
    a = (box[0] + box[2]) / 2
    b = (box[1] + box[3]) / 2
    return np.array([a, b])


def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))
    return dst


def align_image(image, top_left, top_right, bottom_right, bottom_left):
    top_left_point = get_center_bbox(top_left)
    top_right_point = get_center_bbox(top_right)
    bottom_right_point = get_center_bbox(bottom_right)
    bottom_left_point = get_center_bbox(bottom_left)
    source_points = np.float32(
        [top_left_point, top_right_point, bottom_right_point, bottom_left_point]
    )
    crop = perspective_transform(image, source_points)
    return crop


@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    elif dataset_type == "custom":
        class_names = MyDataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    pixel_border = 40

    for i, image_path in enumerate(image_paths):
        start = time.time()
        image_name = os.path.basename(image_path)

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        #COLOR 
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        #INPAINT + CLAHE
        grayimg = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
        clah_inpaint_img = cv2.inpaint(image, mask, 0.1, cv2.INPAINT_TELEA)

        # Detail enhance and create border
        dst = cv2.detailEnhance(clah_inpaint_img, sigma_s=10, sigma_r=0.15)
        dst= cv2.copyMakeBorder(dst, pixel_border, pixel_border, pixel_border, pixel_border, cv2.BORDER_CONSTANT,value=(255,255,255))

        
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

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

        for i in range(len(boxes)):
            for k in range(len(boxes[i])):
                boxes[i][k] -= pixel_border
                if boxes[i][k] < 0:
                    boxes[i][k] = 0

        drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))

        # Crop image
        pair = zip(labels, boxes)
        sort_pair = sorted(pair)
        boxes = [element for _, element in sort_pair]
        labels = [element for element, _ in sort_pair]
        image_cv2 = cv2.imread(image_path)
        cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        if len(boxes) == 4:
            crop = align_image(image_cv2, boxes[0], boxes[1], boxes[2], boxes[3])
        elif len(boxes) == 3:
            # Find fourth missed corner
            thresh = 0
            if 1 not in labels:
                midpoint = np.add(boxes[0], boxes[2]) / 2
                y = int(2 * midpoint[1] - boxes[1][1] + thresh)
                x = int(2 * midpoint[0] - boxes[1][0] + thresh)
                TL = np.array([x, y, x, y])
                crop = align_image(image_cv2, TL, boxes[0], boxes[1], boxes[2])
            elif 2 not in labels:
                midpoint = np.add(boxes[0], boxes[1]) / 2
                y = int(2 * midpoint[1] - boxes[2][1] + thresh)
                x = int(2 * midpoint[0] - boxes[2][0] + thresh)
                TR = np.array([x, y, x, y])
                crop = align_image(image_cv2, boxes[0], TR, boxes[1], boxes[2])
            elif 3 not in labels:
                midpoint = np.add(boxes[2], boxes[1]) / 2
                y = int(2 * midpoint[1] - boxes[0][1] + thresh)
                x = int(2 * midpoint[0] - boxes[0][0] + thresh)
                BR = np.array([x, y, x, y])
                crop = align_image(image_cv2, boxes[0], boxes[1], BR, boxes[2])
            elif 4 not in labels:
                midpoint = np.add(boxes[0], boxes[2]) / 2
                y = int(2 * midpoint[1] - boxes[1][1] + thresh)
                x = int(2 * midpoint[0] - boxes[1][0] + thresh)
                BL = np.array([x, y, x, y])
                crop = align_image(image_cv2, boxes[0], boxes[1], boxes[2], BL)
        else:
            print("Please take a photo again, number of detected corners is:", len(boxes))


        output_dir_crop = os.path.join(images_dir, 'crop')
        cv2.imwrite(os.path.join(output_dir_crop, image_name), crop)



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
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
