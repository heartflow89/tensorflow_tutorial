import tensorflow as tf

# 데이터셋
x_data = [[4,0],[5,1],[4,3],[5,4],[4,6],[5,6],[5,10],[7,1]]
y_data = [[0],[0],[0],[0],[1],[1],[1],[0]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sigmoid function, logisitc function : tf.div(1., 1. + tf.exp(tf.matmul(X,W)+b))
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
# cost, loss
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
# cost minimize
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# hypothesis가 0.5보다 크면 1, 아니면 0
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 정확도
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        
        if step % 100 == 0:
            print(step, cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print(h, '\n', c, '\n', a)