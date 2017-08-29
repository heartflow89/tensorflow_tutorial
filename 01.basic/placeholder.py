import tensorflow as tf

# placeholder : 데이터를 입력 받을 변수(변수 선언, 빈 노드 생성) : 변수를 선언한 후 실행할 때 값을 할당
# tf.placeholder(type, shape, name)
n1 = tf.placeholder(tf.float32)
n2 = tf.placeholder(tf.float32)
n3 = tf.add(n1, n2)

sess = tf.Session()
# feed_dict : placeholder로 생성한 노드에 데이터를 넘겨
print(sess.run(n3, feed_dict={n1:5.0, n2:10.0}))
print(sess.run(n3, feed_dict={n1:[5.0, 10.3], n2:[15.5, 30.2]}))
print(sess.run(n3, feed_dict={n1:[[5.0, 10.3], [3.3, 4.1]], n2:[[15.5, 30.2], [3.2, 5.4]]}))
sess.close()

#X = tf.placeholder(tf.float32, [None, 3])   # None 크기가 정해져있지 않음, 요소3
#x_data = [[1, 2, 3], [4, 5, 6]]