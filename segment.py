import os
import numpy as np
import cv2
import argparse

def segment(images_path, segs_path):

    images = os.listdir(images_path)
    segmentations  = os.listdir(segs_path)

    for im, seg in zip(images, segmentations):

        img = cv2.imread(images_path + im)
        seg = cv2.imread(segs_path + seg)

        cv2.addWeighted(img, 0.5, seg, 0.5, 0.0, img)

        cv2.imshow("img", img)
        cv2.waitKey()

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str)
parser.add_argument("--annotations", type=str)
args = parser.parse_args()

segment(args.images, args.annotations)
