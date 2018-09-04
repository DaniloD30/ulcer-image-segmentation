import argparse
import cv2
import numpy as np
import os
import random

from models import vggsegnet
from loadbatches import imageArray
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--epoch_number", type=int, default=5)
parser.add_argument("--test_images", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--n_classes", type=int)

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

m = vggsegnet.VGGSegnet(n_classes, input_height=input_height, input_width=input_width)
m.load_weights(args.save_weights_path + "." + str(epoch_number))
m.compile(loss='categorical_crossentropy', \
          optimizer= 'adadelta', \
          metrics=['accuracy'])

output_height = m.outputHeight
output_width = m.outputWidth

images = [images_path + f for f in os.listdir(images_path)]

colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(n_classes)]

for imgName in images:

    outName = imgName.replace(images_path, args.output_path)

    X = imageArray(imgName, args.input_width, args.input_height)

    pr = m.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):

        seg_img[:,:,0] += ((pr[:,:] == c) * colors[c][0]).astype('uint8')
        seg_img[:,:,1] += ((pr[:,:] == c) * colors[c][1]).astype('uint8')
        seg_img[:,:,2] += ((pr[:,:] == c) * colors[c][2]).astype('uint8')

    im = cv2.imread(imgName)
    seg_img = cv2.resize(seg_img, (im.shape[1], im.shape[0]))
    cv2.imwrite(outName, seg_img)
