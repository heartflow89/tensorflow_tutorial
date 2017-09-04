import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, :-1]
y_data = xy[:, [-1]]

x_size=len(x_data[0])
y_size=len(y_data[0])

X = tf.placeholder(tf.float32, shape=[None, x_size])
Y = tf.placeholder(tf.float32, shape=[None, y_size])

W = tf.Variable(tf.random_normal([x_size, y_size]), name='weight')
b = tf.Variable(tf.random_normal([y_size]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
        
        if step % 100 == 0:
            print(step, cost_val)
    
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print(h, '\n', '\n', p, '\n', a)