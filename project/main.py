import matplotlib.pyplot as plt
import tensorflow as tf
from create_dataset import *

# ======= Creating dataset from own data =======

data_dir = '../data/datasets'
dataset = create_dataset(data_dir)

# struct of dataset should be next
# <ParallelMapDataset shapes: ((128, 128, 3), (128, 128, 1)), types: (tf.float32, tf.float32)>
print(dataset)


# func in case you need display some elements
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Image', 'Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for image, mask in dataset.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

# =================================================

# ======= Definig AI model  =======

# TODO Olya or Marichka
# look https://www.tensorflow.org/tutorials/images/segmentation#define_the_model

# =================================
