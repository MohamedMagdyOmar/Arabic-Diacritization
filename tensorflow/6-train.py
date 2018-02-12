# we need a way to check if the accuracy is improving overtime during training
# so we will add code that shows how training is progressing
# every 5 steps we will print out current accuracy

import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

testing_data_df = pd.read_csv("sales_data_test.csv", dtype=float)

X_training = training_data_df.drop("total_earnings", axis=1).values
Y_training = training_data_df[["total_earnings"]].values

X_testing = testing_data_df.drop("total_earnings", axis=1).values
Y_testing = testing_data_df[["total_earnings"]].values

X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)

print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))

learning_rate = 0.001
training_epochs = 100
display_step = 5

number_of_inputs = 9
number_of_outputs = 1

layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

with tf.variable_scope('layer_1'):

    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer)
    layer_1_output = tf.nn.relu(tf.matmul(x, weights) + biases)

with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer)
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer)
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer)
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        session.run(optimizer, feed_dict={x: X_scaled_training, Y: Y_scaled_training})

        # every 5 training steps, log our progress
        if epoch % 5 == 0:
            training_cost = session.run(cost, feed_dict={x: X_scaled_training, Y: Y_scaled_training})
            testing_cost = session.run(cost, feed_dict={x: X_scaled_testing, Y: Y_scaled_testing})
            print(epoch, training_cost, testing_cost)

        # print("Training pass: {}".format(epoch))

    print("Training is complete")

    # to check final cost after finish training
    final_training_cost = session.run(cost, feed_dict={x: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={x: X_scaled_testing, Y: Y_scaled_testing})

    # to check final cost after finish training
    print("Final Testing Cost:{}".format(final_training_cost))
    print("Final Testing Cost:{}".format(final_testing_cost))
