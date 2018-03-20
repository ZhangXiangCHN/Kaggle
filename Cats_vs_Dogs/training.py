""" 
coding: utf-8
@author: zhangxiang
"""
import os
import tensorflow as tf
import input_data
import model
import numpy as np

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000
LEARNING_RATE = 0.0001

def run_training():
    train_dir = './data/train/'
    train_log_dir = './logs/train/'

    train, train_label = input_data.get_files(train_dir)

    train_batch, train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, batch_size=BATCH_SIZE,
                                                    capacity=CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, LEARNING_RATE)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 使用try：...except:...finally:...是TF推荐的标准的代码格式，可以获取错误
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 50 == 0:
                print('Step %d, train loss: %.2f, train accuracy:%.2f%%'%(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 200 ==0 or (step+1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('the training is done!')
    finally:
        coord.should_stop()

    coord.join(threads)
    sess.close()


#%% Evaluate one image
# when training, comment the following codes.

#from PIL import Image
#import matplotlib.pyplot as plt
#
#def get_one_image(train):
#    '''Randomly pick one image from training data
#    Return: ndarray
#    '''
#    n = len(train)
#    ind = np.random.randint(0, n)
#    img_dir = train[ind]
#
#    image = Image.open(img_dir)
#    plt.imshow(image)
#    image = image.resize([208, 208])
#    image = np.array(image)
#    return image
#
#def evaluate_one_image():
#    '''Test one image against the saved models and parameters
#    '''
#
#    # you need to change the directories to yours.
#    train_dir = './data/train/'
#    train, train_label = input_data.get_files(train_dir)
#    image_array = get_one_image(train)
#
#    with tf.Graph().as_default():
#        BATCH_SIZE = 1
#        N_CLASSES = 2
#
#        image = tf.cast(image_array, tf.float32)
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image, [1, 208, 208, 3])
#        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#
#        logit = tf.nn.softmax(logit)
#
#        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#
#        # you need to change the directories to yours.
#        logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'
#
#        saver = tf.train.Saver()
#
#        with tf.Session() as sess:
#
#            print("Reading checkpoints...")
#            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#            if ckpt and ckpt.model_checkpoint_path:
#                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                saver.restore(sess, ckpt.model_checkpoint_path)
#                print('Loading success, global_step is %s' % global_step)
#            else:
#                print('No checkpoint file found')
#
#            prediction = sess.run(logit, feed_dict={x: image_array})
#            max_index = np.argmax(prediction)
#            if max_index==0:
#                print('This is a cat with possibility %.6f' %prediction[:, 0])
#            else:
#                print('This is a dog with possibility %.6f' %prediction[:, 1])

if __name__ == "__main__":
    run_training()