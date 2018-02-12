# our training data set has 9 input features, so we will need 9 input neuron network
# then we have 3 layers of neuron network
# first layer has 50 nodes, second layer has 100 nodes, third layer has 50 nodes, and OP layer has 1 node

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# define model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

# define how many input and output neurons
number_of_inputs = 9
number_of_outputs = 1

# define how many neurons in each layer
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# section one: define the layers of the neural network itself

# input layer
# we put each layer in its "variable scope"
# any variables we create within this scope will automatically get a prefix of "input" to their name internally in tf
# "None" tells tf that our NN can mix up batches of any size
# "number_of_inputs" tells tf to expect nine values for each record in the batch

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

# layer 1
# each fully connected layer has the following 3 parts
# "weights" is the value for each connection between each node and the node in previous layer
# "bias" value for each node
# "activation function" that outputs the result of the layer
with tf.variable_scope('layer_1'):

    # "shape", we want to have one weight for each node's connection to each node in the previous layer.
    # "initializer" a good choice for initializing weights is an algorithm called "Xavier Initialization"
    weights = tf.get_variable(name="weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())

    # we need "variables" to store bias value for each node
    # this will be a "variable" instead of a "placeholder", because we want tf to remember the value over time
    # there is one bias value for each node in this layer, so the shape should be the same as the number of nodes in the
    # ... layer
    # we need also to define "initial value" for this variable, so we can pass one of the built-in initializer functions
    # we want the bias values for each node to default to zero
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer)

    # "activation function", multiplying "weights" by "inputs"
    layer_1_output = tf.nn.relu(tf.matmul(x, weights) + biases)

# layer 2
with tf.variable_scope('layer_2'):

    weights = tf.get_variable(name="weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer)
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# layer 3
with tf.variable_scope('layer_3'):

    weights = tf.get_variable(name="weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer)
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):

    weights = tf.get_variable(name="weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer)
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)

# section 2: define the "cost function" of the NN that will measure prediction

with tf.variable_scope('cost'):

    # expected value, it will be "placeholder" node, because it will feed in a new value each time
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    # "cost" function or "loss" function, tells us how wrong the neural network is when trying to predict the correct output
    # mean square of what we expected and the actual(calculated)
    # we want to get avg value of that difference, so we will use "reduce_mean"
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))


# section 3: define the optimization function that will be run to optimize the neural network
with tf.variable_scope('train'):

    # we will use here adam optimizer, that tells tf that whenever we tell it to execute the optimizer, it should run
    # one iteration of AdamOptimizer to make the cost function smaller
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)