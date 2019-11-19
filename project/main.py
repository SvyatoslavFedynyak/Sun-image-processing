import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

sodism_dataset_url = 'ftp://picardweb:ftp@ftp.latmos.ipsl.fr/2014/01/PIC-SOD-20140104_0115-535D-RS-MNM-L1B-SL4R_20140613-O1_D1_C0_R6_G1_F0_X0_P0.fits.gz'
sodism_dataset = tf.keras.utils.get_file(origin=sodism_dataset_url,
                                         fname='sodism_2014')


