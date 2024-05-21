import tensorflow as tf 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

adj_input = tf.constant([[4., 4.]])
adj_output = tf.constant([[2., 2.]])

anomaly_score = tf.reduce_sum(tf.square(adj_input - adj_output))

a = sess.run(anomaly_score)

print(a)