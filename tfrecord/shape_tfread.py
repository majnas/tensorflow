import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io as io

# number of all samples in test or train folders
# in this case test folders
samples_num = 32

tfrecords_filename = 'shape_test.tfrecords'

def read_and_decode(filename_queue, batch_size):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string),
        })

    
    # Convert from a scalar string tensor to a uint8 tensor
    image_raw = tf.decode_raw(features['image_raw'], tf.uint8)
    image_resized = tf.reshape(image_raw, [32, 32])
    
    # Convert from a scalar string tensor to a uint8 tensor
    label_raw = tf.decode_raw(features['label_raw'], tf.uint8)
    label_resized = tf.reshape(label_raw, [1, 2])
   
    
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


    # reading images with batch_size = 8
    # after each fetch we will get 8 images and their labels
    batch_size2 = 8
    num_epochs2 = 1
    filename_queue2 = tf.train.string_input_producer([tfrecords_filename],  num_epochs=num_epochs2)
    image2, label2 = read_and_decode(filename_queue2, batch_size=batch_size2)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session()  as sess:

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        for i in range(num_epochs1*samples_num//batch_size1):        
            batch_x, batch_y = sess.run([image1, label1])
            print(batch_x.shape)
            print('current batch = '+str(i))

            for j in range(batch_size1):
                io.imsave('./dataset_restored1/'+'batch_'+str(i)+'_image_'+str(j)+'.png', batch_x[j, :, :])


        for i in range(num_epochs2*samples_num//batch_size2):        
            batch_x, batch_y = sess.run([image2, label2])
            print(batch_x.shape)
            print('current batch = '+str(i))

            for j in range(batch_size2):
                io.imsave('./dataset_restored2/'+'batch_'+str(i)+'_image_'+str(j)+'.png', batch_x[j, :, :])




        coord.request_stop()
        coord.join(threads)


