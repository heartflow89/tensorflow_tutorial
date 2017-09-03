import tensorflow as tf
import numpy as np

xy = np.loadtxt('test_file.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,:-1]
y_data = xy[:, [-1]]

x_size = len(x_data[0])
y_size = len(y_data[0])

X = tf.placeholder(tf.float32, shape=[None, x_size])
Y = tf.placeholder(tf.float32, shape=[None, y_size])

W = tf.Variable(tf.random_normal([x_size, y_size]), name='weight')
b = tf.Variable(tf.random_normal([y_size]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        hyp_val, cost_val, _ = sess.run([hypothesis, cost, train], feed_dict={X: x_data, Y: y_data})
        
        if step % 100 == 0 :
            print(step, '\n', cost_val, '\n', hyp_val, '\n', sess.run(W), '\n', sess.run(b))
            
    print('=====test=====')
    print(sess.run(hypothesis, feed_dict={X: [[67, 55, 78, 78], [98, 88, 89, 93]]}))