#-*-coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data", one_hot=True)

input = tf.placeholder(tf.float32,[None,28*28])
image = tf.reshape(input,[-1,28,28,1])

#输出
y = tf.placeholder(tf.float32,[None,10])

#卷积层
def conv2d(input,filter):
	return tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='SAME')

#池化层
def pooling(input):
	return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#初始化偏置
def bias(shape):
	return tf.Variable(tf.zeros(shape))

#3*3、步长为1、数量32的卷积核
filter = [3,3,1,32]

filter_conv1 = weight_variable(filter)
b_conv1 = bias([32])

#relu激活函数
h_conv1 = tf.nn.relu(conv2d(image,filter_conv1) + b_conv1)
h_pool1 = pooling(h_conv1)

h_flat = tf.reshape(h_pool1,[-1,14*14*32])

w_fc1 = weight_variable([14*14*32,768])
b_fc1 = bias([768])
h_fc1 = tf.matmul(h_flat,w_fc1) + b_fc1

w_fc2 = weight_variable([768,10])
b_fc2 = bias([10])

y_hat = tf.matmul(h_fc1,w_fc2) + b_fc2

#交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat))

#学习率
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

predict = tf.equal(tf.argmax(y_hat,1),tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(predict,tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(10000):
		batch_x, batch_y = mnist.train.next_batch(50)

		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={input:batch_x, y:batch_y})
			print("step %d,train accuracy %g" %(i,train_accuracy))

		train_step.run(feed_dict={input:batch_x,y:batch_y})

	print("test accuracy %g" %accuracy.eval(feed_dict={input:mnist.test.images, y:mnist.test.labels}))





