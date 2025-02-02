#! -*- coding: utf-8 -*-


import os
from PIL import Image

from utils.file_utils import create_if_not_exists, copy_file
from utils.xml_utils import create_xml_file
from tqdm import tqdm
import json

def reorient_image(im):
    try:
        image_exif = im._getexif()
        image_orientation = image_exif[274]
        if image_orientation in (2,'2'):
            return im.transpose(Image.FLIP_LEFT_RIGHT)
        elif image_orientation in (3,'3'):
            return im.transpose(Image.ROTATE_180)
        elif image_orientation in (4,'4'):
            return im.transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (5,'5'):
            return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (6,'6'):
            return im.transpose(Image.ROTATE_270)
        elif image_orientation in (7,'7'):
            return im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (8,'8'):
            return im.transpose(Image.ROTATE_90)
        else:
            return im
    except (KeyError, AttributeError, TypeError, IndexError):
        return im


class PASCALVOC07(object):

    def __init__(self, trainval_anno, val_anno, test_anno, out_dir, attrs):
        self._trainval_anno = trainval_anno
        self._val_anno = val_anno
        self._test_anno = test_anno
        self._out_dir = out_dir
        self._attrs = attrs

        self._jpegimages_dir = None
        self._imagesets_dir = None
        self._annotations_dir = None
        self._img_idx = 0

    def _build_voc_dir(self):
        self._out_dir = self._out_dir
        create_if_not_exists(os.path.join(self._out_dir, 'Annotations'))
        create_if_not_exists(os.path.join(self._out_dir, 'ImageSets'))
        create_if_not_exists(os.path.join(self._out_dir, 'ImageSets', 'Layout'))
        create_if_not_exists(os.path.join(self._out_dir, 'ImageSets', 'Main'))
        create_if_not_exists(os.path.join(self._out_dir, 'ImageSets', 'Segmentation'))
        create_if_not_exists(os.path.join(self._out_dir, 'JPEGImages'))
        create_if_not_exists(os.path.join(self._out_dir, 'SegmentationClass'))
        create_if_not_exists(os.path.join(self._out_dir, 'SegmentationObject'))
        self._annotations_dir = os.path.join(self._out_dir, 'Annotations')
        self._jpegimages_dir = os.path.join(self._out_dir, 'JPEGImages')
        self._imagesets_dir = os.path.join(self._out_dir, 'ImageSets', 'Main')

    def _create_annotation(self, image_idx, boxes):
        anno_file = os.path.join(self._annotations_dir, "{:06d}.xml".format(image_idx))
        attrs = dict()
        attrs['image_name'] = "{:06d}.jpg".format(image_idx)
        attrs['boxes'] = boxes

        img = Image.open(os.path.join(self._jpegimages_dir, "{:06d}.jpg".format(image_idx)))
        img = reorient_image(img)

        width, height = img.size
        attrs['width'] = str(width)
        attrs['height'] = str(height)
        for k, v in self._attrs.items():
            attrs[k] = v
        create_xml_file(anno_file, attrs)

    def _build_subset(self, start_idx, phase, anno_file, verbose=True, delimiter=' '):

        fout = open(os.path.join(self._imagesets_dir, '{}.txt'.format(phase)), 'w')

        # dictionary_image_id = {}
        n = 0
        with open(anno_file, 'r', encoding = "utf-8") as anno_f:
            for line in tqdm(anno_f):
                line_split = line.strip().split(delimiter)

                # image saved path
                image_path = line_split[0]

                # a ground truth with bounding box
                boxes = []
                for i in range(int((len(line_split) - 1) / 5)):
                    category = line_split[1 + i * 5 + 0]
                    x1 = line_split[1 + i * 5 + 1]
                    y1 = line_split[1 + i * 5 + 2]
                    x2 = line_split[1 + i * 5 + 3]
                    y2 = line_split[1 + i * 5 + 4]
                    boxes.append((category, x1, y1, x2, y2))

                image_idx = start_idx + n
                n += 1
                # copy and rename image by index number
                copy_file(image_path, self._jpegimages_dir, '{:06}.jpg'.format(image_idx))
                # dictionary_image_id[image_path.split("/")[-1]] = '{:06}.jpg'.format(image_idx)
                
                # write image idx to imagesets file
                fout.write('{:06}'.format(image_idx) + '\n')

                # create annotation file
                self._create_annotation(image_idx, boxes)

        fout.close()

        if (phase == "test"):
            fout = open(os.path.join(self._imagesets_dir, 'test_full.txt'), 'w')
            for image_idx in range(1, n + 1):
                fout.write('{:06}'.format(image_idx) + '\n')

            fout.close()

        # with open("../data/dictionary_image.json", "a") as outfile: 
        #     json.dump(dictionary_image_id, outfile)
        #     outfile.close()
            
        return n

    def build(self, start_idx=1, verbose=True):
        self._build_voc_dir()

        n_train = self._build_subset(start_idx, "train", self._trainval_anno, verbose, delimiter = ",")
        n_val = self._build_subset(n_train + start_idx, "val", self._val_anno, verbose, delimiter = ",")
        self._build_subset(n_train + n_val + start_idx, "test", self._test_anno, verbose, delimiter = ",")
