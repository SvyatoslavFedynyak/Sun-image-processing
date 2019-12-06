from PIL import Image
import glob
import os
import tensorflow as tf
from PIL import Image
import cv2 as cv
import numpy as np

dataset_path = '../data/dataset'

def jpg_2_png():
    for mask in glob.glob('{0}/mask/*'.format(dataset_path)):
        im = Image.open(mask)
        new = os.path.basename(mask).replace('jpg', 'png')
        im.save('{0}/mask/{1}'.format(dataset_path, new))

def to_grayscale():
    for mask in glob.glob('{0}/mask/*'.format(dataset_path)):
        img = Image.open(mask)
        img1 = img.convert('L')
        img1.save(mask)

def range_to_one():
    for mask in glob.glob('{0}/mask/*'.format(dataset_path)):
        image = cv.imread(mask)
        hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)

        # Define lower and uppper limits of what we call "brown"
        #brown_lo=np.array([98,48,141])
        #brown_hi=np.array([100,50,143])

        # Mask image to only select browns
        #mask=cv.inRange(hsv,brown_lo,brown_hi)

        # Change image to red where we found brown
        image[np.where((image==[99,49,142]).all(axis=2))] = [127,0,255]
        #image[mask>0]=(127,0,255)

        cv.imwrite(mask,image)

range_to_one()