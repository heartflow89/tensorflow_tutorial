import tensorflow as tf

# 파일을 queue에 쌓음, 파일명을 list로 나열, shuffle은 랜덤하게 섞을 것인지를 의미
filename_queue = tf.train.string_input_producer(['test_file.csv', 'test_file02.csv'], shuffle=True, name='filename_queue')

# reader(파일의 데이터를 읽어옴) 객체 생성 / 연결
# TextLineReader : 파일의 한줄씩 읽어서 리턴
reader = tf.TextLineReader()
# value 파일에서 읽은 값
key, value = reader.read(filename_queue)

# 각 필드의 default값 지정 + 타입 정의하는 역할
record_defaults = [[0.],[0.],[0.],[0.],[0.]]
# 읽어온 값을 xy 노드에 저장
xy = tf.decode_csv(value, record_defaults=record_defaults)
# 데이터를 몇개씩 읽어오면서 학습할지 지정
x_data, y_data = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=5)

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Queue Runner의 쓰레드 관리
    coord = tf.train.Coordinator()
    # Queue Runnser에서 사용할 쓰레드 생성
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    
    for step in range(5001):
        x_batch, y_batch = sess.run([x_data, y_data])
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
        if step % 100 == 0:
            print(step, '\ncost : ', cost_val, '\nhyp : ', hy_val)
    
    # 쓰레드 정지
    coord.request_stop()
    coord.join(threads)