#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/
# base code: TensorFlow Github
# date: 2018-March-26
#-----------------------------------------------------------------------------#

"""
    this code is for reading large scale image datasets, and also using this
    code is easy to handdle images with different sizes.
    shape_maltishape_tfwrite.py for convert all images to tfrecord binary file.
    shape_maltishape_tfread.py reading tfrecord file and restore image dataset.
"""

import tensorflow as tf
from scipy.misc import imsave
import os

# number of all samples in test or train folders
# in this case test folders
samples_num = 20
IMAGE_HEIGHT = 28   # height of images after cropping or padding
IMAGE_WIDTH = 28    # width of images after cropping or padding

tfrecords_filename = 'shape_test.tfrecords'
#tfrecords_filename = 'shape_train.tfrecords'

def read_and_decode(filename_queue, batch_size):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_raw': tf.FixedLenFeature([], tf.string),
                })

    # Convert from a scalar string tensor to a uint8 tensor
    image_raw = tf.decode_raw(features['image_raw'], tf.uint8)
    label_raw = tf.decode_raw(features['label_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([height, width, 1])

    # reshape images to their original sizes
    image_reshaped = tf.reshape(image_raw, image_shape)
    label_resized = tf.reshape(label_raw, [1, 2])

    image_resized = tf.image.resize_image_with_crop_or_pad(image=image_reshaped,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)

    images, labels = tf.train.batch([image_resized, label_resized],
                                    batch_size=batch_size,
                                    capacity=samples_num,
                                    num_threads=2,)

    return images, labels

with tf.Graph().as_default():
    # reading images with batch_size = 4
    # after each fetch we will get 4 images and their labels
    batch_size1 = 4
    num_epochs1 = 1
    filename_queue1 = tf.train.string_input_producer([tfrecords_filename], num_epochs=num_epochs1)
    image1, label1 = read_and_decode(filename_queue1, batch_size=batch_size1)


    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    restored_dataset_dir = './restored_dataset'
    restored_dataset_images_dir =os.path.join(restored_dataset_dir, 'images')
    if not os.path.exists(restored_dataset_images_dir):
        os.makedirs(restored_dataset_images_dir)

    with tf.Session()  as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        for i in range(num_epochs1*samples_num//batch_size1):
            batch_x, batch_y = sess.run([image1, label1])
            print(batch_x.shape)
            print('current batch = '+str(i))

            for j in range(batch_size1):
                restored_dataset_images_filename = os.path.join(restored_dataset_images_dir,
                                                                'batch_'+str(i)+'_image_'+str(j)+'.jpg')
                imsave(restored_dataset_images_filename, batch_x[j, :, :, 0])


        coord.request_stop()
        coord.join(threads)


