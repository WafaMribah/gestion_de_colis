from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import random
import numpy as np
import time
import Data

beginTime = time.time()
input_data=Data.load_data()
input_data=Data.reshape(input_data)



data_placeholder = tf.placeholder(tf.float32, [None, 784])

labels_placeholder = tf.placeholder(tf.int64, [None])

Weights = tf.Variable(tf.zeros([784,10]))

biases = tf.Variable(tf.zeros([10]))

model = tf.matmul(data_placeholder,Weights) + biases

#loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_placeholder, logits=data_placeholder)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model,labels=labels_placeholder))
#loss = tf.reduce_sum(labels_placeholder*tf.log(data_placeholder))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


correct_prediction = tf.equal(tf.argmax(model, 1), labels_placeholder)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  # Initialize variables
  sess.run(tf.global_variables_initializer())


  for i in range(100):

      indices =random.randrange(0, len(input_data['train_images']))
      images_batch = input_data['train_images'][indices]
      labels_batch = input_data['train_labels'][indices]

      if i % 100 == 0:
          train_accuracy = sess.run(accuracy, feed_dict={
          data_placeholder: images_batch, labels_placeholder: labels_batch})
          print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
         # feed the training data to the placeholders of the model
  test_accuracy = sess.run(accuracy, feed_dict={
  data_placeholder: input_data['test_images'],
  labels_placeholder: input_data['test_labels']})
  print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))