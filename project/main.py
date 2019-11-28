import matplotlib.pyplot as plt
import tensorflow as tf
from create_dataset import *

dataset = create_dataset('../data/datasets')


def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

#<ParallelMapDataset shapes: ((128, 128, 3), (128, 128, 1)), types: (tf.float32, tf.float32)>
print(dataset)

for image, mask in dataset.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

