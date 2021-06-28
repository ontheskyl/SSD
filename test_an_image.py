import argparse
import logging
import os

import torch
import torch.utils.data

from ssd.config import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.misc import reorient_image

from PIL import Image
import numpy as np
import cv2


def read_image(image_path):
    image = Image.open(image_path)
    image = reorient_image(image).convert("RGB")
    image = np.array(image)
    return image


def apply_model(model, image, device):
    results_dict = {}
    
    cpu_device = torch.device("cpu")
    with torch.no_grad():
        output = model(image.to(device))

        output = [o.to(cpu_device) for o in output]
    results_dict.update(
        {"test": output[0]}
    )
    return results_dict


def inference(model, image_path, device):

    LABEL = ["", "top_left", "top_right", "bottom_right", "bottom_left"]

    img = read_image(image_path)

    img_info = {"width": img.shape[1], "height": img.shape[0]}
    
    prediction = apply_model(model, img, device)
    boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']

    for i in range(len(boxes)):
        b1 = int(max(boxes[i][0] * img_info["width"] / 320, 0))
        b2 = int(max(boxes[i][1] * img_info["height"] / 320, 0))
        b3 = int(min(boxes[i][2] * img_info["width"] / 320, img_info["width"]))
        b4 = int(min(boxes[i][3] * img_info["height"] / 320, img_info["height"]))
        img = cv2.rectangle(img, (b1, b2), (b3, b4), (255, 0, 0), 2)
        img = cv2.putText(img, "{}".format(LABEL[labels[i]]), (b1, b2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, (0, 0, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, "{}".format(round(float(scores[i]), 2)), (b1, b2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imwrite("prediction.jpg", img)


@torch.no_grad()
def do_evaluation(cfg, image_path, model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    
    inference(model, image_path, device)

def make_prediction(cfg, image_path, model_path, ckpt):
    logger = logging.getLogger("SSD.inference")

    model = build_detection_model(cfg)
    checkpointer = CheckPointer(model, save_dir=model_path, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    do_evaluation(cfg, image_path, model)


def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--image-path",
        help = "The path of the image for testing",
        default="test_img.jpg",
        type = str,
    )
    parser.add_argument(
        "--model-path",
        help = "The path of the model",
        default="model",
        type = str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    image_path = args.image_path
    model_path = args.model_path

    make_prediction(cfg, image_path, model_path, ckpt=args.ckpt)


if __name__ == '__main__':
    main()
