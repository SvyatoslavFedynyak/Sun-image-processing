import tensorflow as tf
import pathlib
import glob
import os

# convert source files to dataset


def create_dataset(datasets_dir):

    # directory with datasets
    data_dir = pathlib.Path(datasets_dir)

    # take files and masks paths and combine in format of "{imade_path|mask_path}"
    objects = []
    for item in glob.glob('{0}/image/*'.format(data_dir)):
        item = os.path.basename(item)
        objects.append('{0}/image/{1}|{0}/mask/{1}'.format(data_dir, item))

    # creates dataset
    list_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(objects))

    # normalize images
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

    # map func for dataset (opend image for path and add as element)
    def process_path(file_path):
        parts = tf.strings.split(file_path, '|')
        img = tf.io.read_file(parts[0])
        img = decode_img(img)
        mask = tf.io.read_file(parts[1])
        mask = decode_img(mask)
        return img, mask

    # map dataset (invokes map func for every element)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
   
    return labeled_ds
