## Multi-variable Linear Regression ##
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


x1 = [73., 93., 89., 96., 73.]
x2 = [80., 88., 91., 98., 66.]
x3 = [75., 93., 90., 100., 70.]
y = [152., 185., 180., 196., 142.]

w1 = tf.Variable(10.)
w2 = tf.Variable(10.)
w3 = tf.Variable(10.)
b = tf.Variable(10.)

learning_rate = 0.000001

for i in range(1000+1):
    with tf.GradientTape() as tape:
        hypothesis = (w1 * x1) + (w2 * x2) + (w3 * x3) +b
        cost = tf.reduce_mean(tf.square(hypothesis - y))
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1,w2,w3,b])

    w1.assign_sub(learning_rate * w1_grad)
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5}|{:12.4f}".format(i, cost.numpy()))


