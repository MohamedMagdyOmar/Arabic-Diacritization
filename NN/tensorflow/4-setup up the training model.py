import tensorflow as tf

with tf.Session() as session:

    # run the global variable initializer to initialize all variables and layers to their default values
    session.run(tf.global_variables_initializer())

    # here we run our ptimizer function over and over, either for a certain number of iterations
    # ... or unitll it hits an accuracy level we want
    for epoch in range(training_epochs):

        # feed in the training data and do one step of  neural network training
        session.run(optimizer, feed_dict={x: X_scaled_training, y: y_scaled_training})

        print("Training pass: {}".format(epoch))

    print("Training is complete")