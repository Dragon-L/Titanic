import tensorflow as tf
from mannual.deep_learning import initialize_parameters


def initialize_parameters(layer_dim):
    parameters = {}
    for l in range(1, len(layer_dim)):
        parameters['W' + str(l)] = tf.get_variable('W' + str(l), [layer_dim[l], layer_dim[l - 1]],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dim[l], 1], initializer=tf.zeros_initializer())
    return parameters


def forward_propagation(X, parameters, layer_len, activation):
    cache = {}
    cache['A0'] = X
    for l in range(1, layer_len):
        cache['Z' + str(l)] = tf.add(tf.matmul(parameters['W' + str(l)], cache['A' + str(l-1)]), parameters['b' + str(l)])
        if activation == 'relu':
            cache['A' + str(l)] = tf.nn.relu(cache['Z' + str(l)])

    return cache['Z' + str(layer_len - 1)]


def compute_cost(Y_train, Zl):
    logits = tf.transpose(Zl)
    labels = tf.transpose(Y_train)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

def training_parameters(X_train, Y_train, layer_dim, activation, parameters_origin, learning_rate, num_iterations = 1000):
    X = tf.placeholder(tf.float32, shape=[layer_dim[0], None])
    Y = tf.placeholder(tf.float32, shape=[1, None])
    parameters = initialize_parameters(layer_dim)

    Zl = forward_propagation(X, parameters, len(layer_dim), activation)
    cost = compute_cost(Y, Zl)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_iterations):
            _, current_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            if i % 1000 == 0:
                print("Cost after iterations %i: %f" % (i, current_cost))

        parameters = sess.run(parameters)

        half_value = tf.constant(0.5, shape=[Y_train.shape[0], Y_train.shape[1]])
        true_value = tf.constant(1.0, shape=[Y_train.shape[0], Y_train.shape[1]])
        result = tf.nn.sigmoid(Zl)
        Y_predict = tf.greater_equal(result, half_value)
        Y_target = tf.equal(Y, true_value)

        correct_prediction = tf.equal(Y_predict, Y_target)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        print('Train Accuracy:', accuracy.eval({X: X_train, Y: Y_train}))

    return parameters