import tensorflow as tf
import numpy as np

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2], name='x_input')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='y_input')

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    
    # 기록할 노드 선택
    w1_hist = tf.summary.histogram('weights1', W1)
    b1_hist = tf.summary.histogram('biases1', b1)
    layer1_hist = tf.summary.histogram('layer1', layer1)
    
with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    
    w2_hist = tf.summary.histogram('weights2', W2)
    b2_hist = tf.summary.histogram('biases2', b2)
    hypotehsis_hist = tf.summary.histogram('hypothesis', hypothesis)
    
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)
    
with tf.name_scope("train") as scope:
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_sca = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    # summary 병합
    merged_summary = tf.summary.merge_all()
    # summary를 기록할 위치 지정
    writer = tf.summary.FileWriter("./logs/xor_logs")
    writer.add_graph(sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        # summary 실행 및 파일 기록
        summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)
        
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run([W1, W2]))
            
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print(h, c, a)