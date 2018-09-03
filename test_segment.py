import os
import numpy as np
import cv2
import argparse

def segment(images_path, segs_path, result_path):

    images = os.listdir(images_path)
    images.sort()
    segmentations  = os.listdir(segs_path)
    segmentations.sort()

    for im, seg in zip(images, segmentations):

        img = cv2.imread(images_path + im)
        seg = cv2.imread(segs_path + seg)

        cv2.addWeighted(img, 0.5, seg, 0.5, 0.0, img)

        cv2.imwrite(result_path + im, img)

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str)
parser.add_argument("--annotations", type=str)
parser.add_argument("--result", type=str)
args = parser.parse_args()

segment(args.images, args.annotations, args.result)
