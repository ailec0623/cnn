# -*- coding: utf-8 -*-
#tersorflow 新手教程神经网络
from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf
import time

#初始化权重和偏移量
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#定义卷积和池化操作
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


start = time.clock()
#获取数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
with tf.device('/gpu:0'):
  #设置数据占位符
  x = tf.placeholder("float", shape=[None, 784])
  y_ = tf.placeholder("float", shape=[None, 10])

  #第一次卷积操作
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])
with tf.device('/gpu:0'):
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)


#加入全连接层
W_fc1 = weight_variable([14 * 14 * 32, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool1, [-1, 14*14*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#加入输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
keep_prob = tf.placeholder("float")
y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


#计算交叉熵，循环训练，输出准确度
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(tf.initialize_all_variables())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
elapsed = (time.clock() - start)
print elapsed