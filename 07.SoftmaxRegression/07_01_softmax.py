import tensorflow as tf

x_data = [[1,2,1,1], [2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
# one-hot
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

logits = tf.matmul(X, W) + b
# softmax : tf.nn.softmax = exp(logits) / reduce_sum(exxp(logits), dim)
hypothesis = tf.nn.softmax(logits)

# cross entropy cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis)))
# cost minimize
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    # 초기화
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, cost_val = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)
            
    print('=======test========')
    
    a = sess.run(hypothesis, feed_dict={X: [[1,3,4,3]]})
    print(a, sess.run(tf.argmax(a, 1)))
    
    b = sess.run(hypothesis, feed_dict={X: [[1,1,2,2]]})
    print(b, sess.run(tf.argmax(b, 1)))
    
    all = sess.run(hypothesis, feed_dict={X: [[2,1,3,2], [3,4,5,6]]})
    print(all, sess.run(tf.argmax(all, 1)))