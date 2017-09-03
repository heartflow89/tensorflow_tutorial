import tensorflow as tf

x_data = [[70., 80., 75., 88.],
          [60., 59., 73., 77.],
          [77., 86., 90., 95.],
          [95., 95., 100., 98.],
          [86., 87., 90., 88.]]
y_data = [[80.], [71.], [90.], [98.], [92.]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = opt.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, _, hyp_val = sess.run([cost, train, hypothesis], feed_dict = {X : x_data, Y : y_data})
        if step % 40 == 0:
            print(step, cost_val, '\n', hyp_val)
    
    print('======test=====')
    print(sess.run(hypothesis, feed_dict={X : [[70, 81, 79, 83], [80, 85, 86, 88]]}))