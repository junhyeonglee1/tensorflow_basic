# cost 함수 ##
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy
x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]

w = tf.Variable(2.9)
b = tf.Variable(0.5)
learning_rate = 0.01
## Gradient descent ##
for i in range(100+1):
    with tf.GradientTape() as tape:
        hypothesis = w * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    w_grad, b_grad = tape.gradient(cost,[w,b])
    w.assign_sub(learning_rate * w_grad) #w,b 업데이트
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, w.numpy(), b.numpy(), cost))


