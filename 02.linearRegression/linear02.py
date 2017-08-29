import tensorflow as tf

X = tf.placeholder(tf.float32, shape = [None])
Y = tf.placeholder(tf.float32, shape = [None])

W = tf.Variable(tf.random_normal([1]), name='weigth')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설 정의
hypothesis = X * W + b

# cost 정의
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# cost 최적화 / 경사하강법
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = opt.minimize(cost)

# Session 생성(with ~ as 를 이용하면 close를 하지 않아도)
with tf.Session() as sess:
    # Variable 초기화
    sess.run(tf.global_variables_initializer())
    # 최적화 반복 수행(cost를 최소화, W와 b의 값을 학습을 하면서 수정)
    for step in range(501):
        cost_val, _ = sess.run([cost, train], feed_dict = {X : [1, 2, 3, 4, 5], Y: [10, 18, 28, 35, 45]})
        if step % 20 == 0:
            print(step, cost_val, sess.run(W), sess.run(b))
    
    print('=======모델 테스트=======')
    # hypothesis 실행, X의 값을 넣어줘서 잘 작동하는지 확인
    print(sess.run(hypothesis, feed_dict={X : [10]}))
    print(sess.run(hypothesis, feed_dict={X : [12, 15]}))