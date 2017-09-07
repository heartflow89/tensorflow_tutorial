import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# MNIST 데이터 로드
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# 0 ~ 9
classes = 10
# 28 * 28 = 784
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, classes])

W = tf.Variable(tf.random_normal([784, classes]), name='weight')
b = tf.Variable(tf.random_normal([classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# epoch : 데이터를 몇번 학습시킬지
#         (예를 들어 100개의 데이터가 있을경우 100개를 한번 모두 학습하면 1epoch)
# batch : 데이터의 양이 많은 경우 조금씩 데이터를 메모리에 올리면서 학습
#         (예를 들어 batch가 20인 경우 5번 반복하면 1epoch)
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs): # 총 15번 반복 학습한다는 의미
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch): # 1epoch
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch
        print('epoch :', epoch + 1, '\tcost :', avg_cost)
        
    print('=====학습 종료!=====')
    # 학습결과 테스트
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    
    for i in range(3):
        r = random.randint(0, mnist.test.num_examples - 1)
        print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print("Prediction: ", sess.run(prediction, feed_dict={X: mnist.test.images[r:r + 1]}))
        
        plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()