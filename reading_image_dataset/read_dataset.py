"""
Example TensorFlow script for reading dataset.
Based on PyTorch example from Justin Johnson
(https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c)
Required packages: tensorflow (>=v1.2)
```

The whole data is stored on disk; each m category has its own folder on disk
and n images for that category are stored as *.png files in the category folder.
In other words, the directory structure looks something like this:

dataset
   '
   '---- train
   '       '
   '       '---- category_1
   '       '       '
   '       '       '---- image_1.png
   '       '       '---- image_2.png
   '       '       '---- ....
   '       '       '---- image_n.png
   '       '
   '       '---- category_2
   '       '       '
   '       '       '---- image_1.png
   '       '       '---- image_2.png
   '       '       '---- ....
   '       '       '---- image_n.png
   '       '
   '       '----  ...
   '       '       
   '       '       
   '       '
   '       '---- category_m
   '               '
   '               '---- image_1.png
   '               '---- image_2.png
   '               '---- ....
   '               '---- image_n.png
   '       
   ' --- test 
           '
           '---- category_1
           '       '
           '       '---- image_1.png
           '       '---- image_2.png
           '       '---- ....
           '       '---- image_n.png
           '
           '---- category_2
           '       '
           '       '---- image_1.png
           '       '---- image_2.png
           '       '---- ....
           '       '---- image_n.png
           '
           '----  ...
           '       
           '       
           '
           '---- category_m
                   '
                   '---- image_1.png
                   '---- image_2.png
                   '---- ....
                   '---- image_n.png
       

"""



import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='dataset/train')
parser.add_argument('--test_dir', default='dataset/test')
parser.add_argument('--batch_size', default=4   , type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--num_epochs1', default=1, type=int)
parser.add_argument('--restore_dir', default='./dataset_restored/')

n_train = 64
n_test = 16

def list_images(directory):
    """
    Get all the images and labels in directory/label/*.png
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and test get them in the same order
    labels.sort()
    

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))
    
    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i
        

    labels = [label_to_int[l] for l in labels]

    return filenames, labels



def main(args):
    # Get the list of filenames and corresponding list of labels for training et test
    train_filenames, train_labels = list_images(args.train_dir)
    test_filenames, test_labels = list_images(args.test_dir)    
    assert set(train_labels) == set(test_labels),\
           "Train and test labels don't correspond:\n{}\n{}".format(set(train_labels),
                                                                   set(test_labels))
    
    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        # Decode the image from png format
        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            return image, label

        # Preprocessing (for both training and testing):
        # Horizontally flip the image with probability 1/2
        def _preprocess(image, label):
            flip_image = tf.image.random_flip_left_right(image)
            return flip_image, label


        # ----------------------------------------------------------------------
        # DATASET CREATION using tf.contrib.data.Dataset
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

        # The tf.contrib.data.Dataset framework uses queues in the background to feed in
        # data to the model.
        # We initialize the dataset with a list of filenames and labels, and then apply
        # the preprocessing functions described above.
        # Behind the scenes, queues will load the filenames, preprocess them with multiple
        # threads and apply the preprocessing in parallel, and then batch the data

        # Training dataset
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(_parse_function,
                                          num_threads=args.num_workers, 
                                          output_buffer_size=args.batch_size)
        train_dataset = train_dataset.map(_preprocess,
                                          num_threads=args.num_workers, 
                                          output_buffer_size=args.batch_size)
        
        train_dataset = train_dataset.shuffle(buffer_size=n_train)  # don't forget to shuffle
        train_dataset = train_dataset.repeat()
        batched_train_dataset = train_dataset.batch(args.batch_size)

        # Test dataset
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_filenames, test_labels))
        test_dataset = test_dataset.map(_parse_function,
                                        num_threads=args.num_workers,
                                        output_buffer_size=args.batch_size)
        test_dataset = test_dataset.map(_preprocess,
                                        num_threads=args.num_workers, 
                                        output_buffer_size=args.batch_size)
        batched_test_dataset = test_dataset.batch(args.batch_size)

        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(test_init_op)   for 1 epoch on the test set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `test_dataset` here, because they are compatible.
        train_iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                                 batched_train_dataset.output_shapes)
        train_images, train_labels = train_iterator.get_next()
        train_init_op = train_iterator.make_initializer(batched_train_dataset)


        test_iterator = tf.contrib.data.Iterator.from_structure(batched_test_dataset.output_types,
                                                                 batched_test_dataset.output_shapes)
        test_init_op = test_iterator.make_initializer(batched_test_dataset)


    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    with tf.Session(graph=graph) as sess:

        # Update only the last layer for a few epochs.
        for epoch in range(args.num_epochs1):
            # Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)
            for itr in range(n_train//args.batch_size):
                # fetch new batch
                batch_x, batch_y = sess.run([train_images, train_labels])
                batch_x = batch_x/255
                
                print(batch_x.shape)
                for j in range(args.batch_size):
                    im = batch_x[j, :, :, :]
                    plt.imsave(args.restore_dir + 'batch_'+str(epoch)+str(itr)+'_image_'+str(j)+'.png', im)

if __name__ == '__main__':
    args = parser.parse_args()
    
    if not os.path.exists(args.restore_dir):
        os.makedirs(args.restore_dir)
        
    main(args)
    