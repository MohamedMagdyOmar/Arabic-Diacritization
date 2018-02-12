# tf enables us to visualize what is happening during training through "tensorboard"
# it enables us to track accuracy of our model during training
# in this lecture we will log "scaler" value that logs "cost" function over time

# in tf we log values by creating special operations in our graph called "summary operations"
# this operation takes a value and create log data in a format that tfboard understand it

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

print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0],
                                                                                      Y_scaler.min_[0]))

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

    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.
                              xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer)
    layer_1_output = tf.nn.relu(tf.matmul(x, weights) + biases)

with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.
                              xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer)
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.
                              xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer)
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.
                              xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer)
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# create a summary operation to log the progress of the network
with tf.variable_scope('logging'):
    # this will represent the value we are logging
    # below you are only logging scaler value(cost), but tf provide you the ability to log complex objects such as
    # ... pictures, histograms, and even sound files
    # we can run this node by calling "session.run" like any other node in our graph
    # but what if we have a lot of metrics, will we need session.run for every one? this is tedious.
    # tf provide us with different way called "tf.summary.merge_all"
    tf.summary.scalar('current_cost', cost)

    # when you run this special node, it will automatically execute all summary nodes in our graph, without listing all
    summary = tf.summary.merge_all()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # create log file writer to record training progress.
    # we'll store training and testing data separately.
    training_writer = tf.summary.FileWriter("./logs/training", session.graph)
    testing_writer = tf.summary.FileWriter("./logs/testing", session.graph)

    for epoch in range(training_epochs):
        session.run(optimizer, feed_dict={x: X_scaled_training, Y: Y_scaled_training})

        if epoch % 5 == 0:
            # we need to add a new call to run our new summary operations here
            # instead of adding 2 new lines, we can just update these 2 lines
            # tf can run more than 1 operation in same "session.run"
            training_cost, training_summary = session.run([cost, summary], feed_dict={x: X_scaled_training,
                                                                                      Y: Y_scaled_training})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={x: X_scaled_testing,
                                                                                    Y: Y_scaled_testing})

            # last step is to write the current training status to the log files(which we can visualize using tfboard)

            # represent x-axis
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

            print(epoch, training_cost, testing_cost)

    print("Training is complete")

    final_training_cost = session.run(cost, feed_dict={x: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={x: X_scaled_testing, Y: Y_scaled_testing})

    print("Final Testing Cost:{}".format(final_training_cost))
    print("Final Testing Cost:{}".format(final_testing_cost))

# finally we run tensorboard using terminal
# tensorboard --logdir=tensorflow/logs
# use generated url to open tensorboard
# https://stackoverflow.com/questions/44175037/cant-open-tensorboard-0-0-0-06006-or-localhost6006
