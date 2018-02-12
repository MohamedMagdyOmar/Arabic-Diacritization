import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = tf.placeholder(tf.float32, name='x')
Y = tf.placeholder(tf.float32, name='y')
addition = tf.add(X, Y, name='result')


with tf.Session() as session:
    result = session.run(addition, feed_dict={X: [1, 2], Y: [2, 5]})
    print(result)
