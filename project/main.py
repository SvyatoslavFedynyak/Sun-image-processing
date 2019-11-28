#import sys
#root_dir = '/home/svyatoslav/projects/git-proj/Sun-image-processing'
#sys.path.insert(0, root_dir)

import create_dataset

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

for image in dataset.take(1):
  sample_image = image
display([sample_image])
