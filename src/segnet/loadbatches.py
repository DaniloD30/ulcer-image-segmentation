import cv2
import itertools
import numpy as np
import os

def imageArray(path, width, height, imgNorm="sub_mean", ordering='channels_first'):

    try:
        img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":

            img = cv2.resize(img, (width, height)).astype(np.float32) / 127.5 - 1

        elif imgNorm == "sub_mean":

            img = cv2.resize(img, (width, height)).astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68

        elif imgNorm == "divide":

            img = cv2.resize(img, (width, height)).astype(np.float32) / 255.0

    except Exception as e:

        print(path, e)

        img = np.zeros((height, width, 3))

    if ordering == 'channels_first':

        img = np.rollaxis(img, 2, 0)

    return img

def segmentationArray(path, nClasses, width, height):

    seg_labels = np.zeros((height, width, nClasses))

    try:

        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:,:,0]

        for c in range(nClasses):

            seg_labels[:,:,c] = (img == c).astype(int)

    except Exception as e:

        print(e)

    seg_labels = np.reshape(seg_labels, (width * height, nClasses))

    return seg_labels

def segmentGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height, output_width):

    assert images_path[-1] == '/' and segs_path[-1] == '/'

    images = [images_path + f for f in os.listdir(images_path)]

    segmentations = [segs_path + f for f in os.listdir(images_path) if os.path.isfile(segs_path + f)]

    zipped = itertools.cycle(zip(images, segmentations))

    while True:

        X = []
        Y = []

        for _ in range(batch_size):

            img, seg = next(zipped)

            X.append(imageArray(img, input_width, input_height))
            Y.append(segmentationArray(seg, n_classes, output_width, output_height))

            yield np.array(X), np.array(Y)
