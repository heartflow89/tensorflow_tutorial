import tensorflow as tf

n1 = tf.constant(10, tf.int32)    # constant는 상수를 의미, tf.float32 는 인코딩 타입을 의미
n2 = tf.constant(15)
n3 = tf.add(n1, n2)     # n1, n2 상수 텐서로 그래프 생성 / 모델 구성

print(n3)   # 정상적으로 실행이 되지 않음 Session 객체 생성 후 실행 필요

sess = tf.Session()     # Session 객체 생성
print("n3 :", sess.run(n3))   # sess.run() : 그래프(변수, 수식 등) 실행

sess.close()    # Session 객체 닫음



#node1 = tf.constant(3.0, tf.float32) # 3.0 노드 생성, 인코딩 타
#node2 = tf.constant(4.0)
#node3 = tf.add(node1, node2)    # node3 = node1 + node2

#sess = tf.Session()
#print("sess.run(node1, node2):", sess.run([node1, node2]))
#print("sess.run(node3)", sess.run(node3))