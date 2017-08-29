import tensorflow as tf

# 데이터셋(학습시킬 데이터)
x_data = [0.0, 1.0, 4.0, 6.0]
y_data = [1.0, 1.5, 3.0, 4.0]

# Variable 노드(변수) 생성 : 기계가 학습하면서 변경시킬 변수
# tf.random_noraml([1]) : 1차원의 정규분포 난수 생성
W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

# 가설 정의
hypothesis = x_data * W + b

# cost 정의
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# 경사 하강법(최적화 시키는 방법) / learning_rate는 학습률을 의미(작을수록 오래 학습)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# cost 최소화(minimize)
train = opt.minimize(cost)

# 세션 객체 생성
sess = tf.Session()
# Variable 사용 시 실행할때 변수 값을 반드시 초기화
sess.run(tf.global_variables_initializer())

for step in range(1001):
    # train 노드 실행(즉, 반복하는 동안 최적화 및 cost 최소화)
    sess.run(train)
    if step % 40 == 0:
        print(step, 'cost =', sess.run(cost), 'W =', sess.run(W), 'b =' , sess.run(b))

sess.close()