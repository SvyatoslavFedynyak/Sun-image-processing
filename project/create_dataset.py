import numpy as np
import pathlib
import os
import tensorflow as tf
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

# convert source files to dataset


def create_dataset(datasets_dir):
    data_dir = pathlib.Path(datasets_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    CLASS_NAMES = np.array(
        [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

    BATCH_SIZE = 2
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)


    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == CLASS_NAMES


    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


    def process_path(file_path):
        label = get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label


    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for image, label in labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())


    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds
    return prepare_for_training(labeled_ds)
