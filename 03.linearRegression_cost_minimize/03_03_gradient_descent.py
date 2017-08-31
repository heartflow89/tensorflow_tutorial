import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name = 'weight')

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = opt.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(30):
        _, cost_val = sess.run([train, cost], feed_dict = {X:[1,2,3], Y:[1,2,3]})
        print(step, cost_val, sess.run(W))