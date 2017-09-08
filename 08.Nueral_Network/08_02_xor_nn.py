import tensorflow as tf
import numpy as np

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([2, 3]))
b1 = tf.Variable(tf.random_normal([3]))
W2 = tf.Variable(tf.random_normal([3, 2]))
b2 = tf.Variable(tf.random_normal([2]))
W3 = tf.Variable(tf.random_normal([2, 1]))
b3 = tf.Variable(tf.random_normal([1]))

hid_1 = tf.sigmoid(tf.matmul(X, W1) + b1)
hid_2 = tf.sigmoid(tf.matmul(hid_1, W2) + b2)
hypothesis = tf.sigmoid(tf.matmul(hid_2, W3) + b2)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.cast(hypothesis > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        c, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        
        if step % 100 == 0:
            print(c)
    
    print('===학습 종료===')
    print(sess.run([hypothesis, prediction, accuracy], feed_dict={X: x_data, Y: y_data}))