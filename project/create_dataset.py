import numpy as np
import pathlib
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import glob

AUTOTUNE = tf.data.experimental.AUTOTUNE

# convert source files to dataset

def create_dataset(datasets_dir):
    data_dir = pathlib.Path(datasets_dir)

    objects = []
    objects.append('../data/datasets/image/1.jpg|../data/datasets/mask/1.jpg')
    objects.append('../data/datasets/image/2.jpg|../data/datasets/mask/2.jpg')

    images = glob.glob ('../data/datasets/image/*')
    masks = glob.glob ('../data/datasets/mask/*')

    list_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(objects))

    #<ParallelMapDataset shapes: ((128, 128, 3), (128, 128, 1)), types: (tf.float32, tf.float32)>

    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


    def process_path(file_path):
        parts = tf.strings.split(file_path, '|')
        img = tf.io.read_file(parts[0])
        img = decode_img(img)
        mask = tf.io.read_file(parts[1])
        mask = decode_img(mask)
        return img, mask


    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
   
    return labeled_ds
