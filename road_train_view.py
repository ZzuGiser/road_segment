"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    python road.py train --dataset=D:\360download\code_targetdetection\Mask_RCNN-master\samples\road --weights=coco


    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""
from PIL import Image
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.io
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
TRAIN_IMAGES_NAME = "train"
TRAIN_IMAGES_PATH = os.path.join(ROOT_DIR, TRAIN_IMAGES_NAME)
IMAGES_OUT_NAME = "result"
IMAGES_OUT_PATH = os.path.join(ROOT_DIR, IMAGES_OUT_NAME)


class ViewDataset(object):

    def load_building(self, dataset_dir=TRAIN_IMAGES_PATH, out_path=IMAGES_OUT_PATH):
        # Train or validation dataset?
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            try:
                rects = [r['shape_attributes'] for r in a['regions']]
                image_path = os.path.join(dataset_dir, a['filename'])
                img_out_path = os.path.join(out_path, a['filename'])
                image = cv2.imread(image_path)
                p = rects[0]
                x, y, x1, y1 = int(p['x']), int(p['y']), int(p['x'] + p['width']), int(p['y'] + p['height'])
                cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)
                cv2.imwrite(img_out_path, image)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    output_pack = '{:%Y%m%d_%H%M}_road_view'.format(datetime.datetime.now())
    dataset_dir = TRAIN_IMAGES_PATH
    out_path = os.path.join(IMAGES_OUT_PATH, output_pack)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    view_dataset = ViewDataset()
    view_dataset.load_building(dataset_dir=dataset_dir, out_path=out_path)
