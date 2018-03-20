""" 
coding: utf-8
@author: zhangxiang
"""
import tensorflow as tf
import numpy as np
import os

# train_dir id the path where the training images are
train_dir = './data/train/'

def get_files(file_dir):
    """
    Args:
        file_dir:file directory
    Returns:
        list of images and labels
    """
    cats = []
    dogs = []
    labels_cat = []
    labels_dog = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            labels_cat.append(0)
        else:
            dogs.append(file_dir + file)
            labels_dog.append(1)

    # merge data
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((labels_cat, labels_dog))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)    # shuffle the data

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    Args:
        image, label: list type
        image_W, image_H: the wide and height of image
        batch_size: batch_size
        capacity: the maximun elements in queue
    returns:
        image_batch and label_batch: 4D tensor for image_batch [batch_size, image_W, image_H, channels], tf.float32 ,
                                                            when image is RGB , channels equal to 3
                                     1D tensor for label_batch [batch_size], tf.int32()
    """
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make a queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])   # dropped was ok
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

# test
# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208
#
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    try:
#        while not coord.should_stop() and i<1:
#
#            img, label = sess.run([image_batch, label_batch])
#
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)