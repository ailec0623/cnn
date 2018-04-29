# -*- coding: utf-8 -*-
#tersorflow 新手教程神经网络
from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf
import time


start = time.time()
#获取数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])

  #设置权重和偏移量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#定义模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

#计算交叉熵
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
keep_prob = tf.placeholder("float")
#初始化变量
init = tf.initialize_all_variables()

#设置会话并启动
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

#循环训练
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#计算准确性
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
elapsed = (time.time() - start)
print elapsed