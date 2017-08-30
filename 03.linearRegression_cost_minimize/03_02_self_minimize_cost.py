import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name = 'weight')
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(30):
        sess.run(update, feed_dict={X : [1,2,3], Y : [1,2,3]})
        print(step, sess.run(cost, feed_dict={X : [1,2,3], Y: [1,2,3]}), sess.run(W))