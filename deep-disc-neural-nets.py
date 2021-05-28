import numpy as np

# Deep Discriminative Neural Networks [week 5]

ERR_MSG = 'Please use proper activation function name. (relu,' \
          'lrelu,tanh,heaviside'


# Returns values after applying activation function.
# Params: net - a 2D numpy ndarray, activation - activation function, kwargs to
# specify additional value for specific activation function.
def calc_activation(net, activation, **kwargs):
    return {
        'relu': np.array([apply(row, relu, **kwargs) for row in net]),
        'lrelu': np.array([apply(row, lrelu, **kwargs) for row in net]),
        'tanh': np.array([apply(row, tanh, **kwargs) for row in net]),
        'heaviside': np.array([apply(row, heaviside, **kwargs) for row in net]),
    }.get(activation, ERR_MSG)


def apply(row, activation_fun, **kwargs):
    result = []
    for value in row:
        result.append(activation_fun(value, **kwargs))
    return result


def relu(x, **kwargs):
    return x if x >= 0 else 0


def lrelu(x, a=0.1, **kwargs):
    return x if x >= 0 else a * x


def tanh(x, **kwargs):
    return np.tanh(x)


def heaviside(x, threshold=0.1, **kwargs):
    activation = x - threshold
    if activation > 0:
        return 1
    elif activation == 0.0:
        return 0.5
    else:
        return 0


# Returns samples after batch normalization.
# Parameters: samples - collections of samples in which every sample needs to
# have the same shape, beta, gamma, epsilon are parameters of the batch
# normalization
def batch_normalization(samples, beta, gamma, epsilon):
    output = []
    shape = samples[0].shape

    means = np.zeros(shape)
    variances = np.zeros(shape)
    for y in range(shape[1]):
        for x in range(shape[0]):
            values_at_x_y = np.array([sample[y][x] for sample in samples])
            mean = np.mean(values_at_x_y)
            variance = np.var(values_at_x_y)

            means[y][x] = mean
            variances[y][x] = variance

    for sample in samples:
        b_normalized_sample = np.zeros(shape)
        for y in range(shape[1]):
            for x in range(shape[0]):
                value = sample[y][x]
                b_normalized_sample[y][x] = calc_b_norm(beta, epsilon, gamma,
                                                        means[y][x], value,
                                                        variances[y][x])
        output.append(b_normalized_sample)

    return output


# Returns batch normalization for a single value
def calc_b_norm(beta, epsilon, gamma, mean, value, variance):
    return beta + gamma * ((value - mean) / np.sqrt(variance + epsilon))


if __name__ == '__main__':
    # Task4 The following array show the output produced by a mask in a
    # convolutional
    # layer of a CNN
    # net =  [  1  0.5  0.2 ]
    #        [ −1 −0.5 −0.2 ]
    #        [0.1 −0.1  0   ]
    #
    # Calculate the values produced by the application of the following
    # activation functions:
    # a. ReLU,
    # b. LReLU when a=0.1,
    # c. tanh,
    # d. Heaviside function where each neuron has a threshold of 0.1
    # (define H(0) as 0.5).
    net = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
    print('input:')
    print(net)

    print('relu:')
    print(calc_activation(net, 'relu'))

    print('lrelu with a=0.1:')
    print(calc_activation(net, 'lrelu', a=0.1))

    print('tanh:')
    print(calc_activation(net, 'tanh'))

    print('heaviside with threshold = 0.1:')
    print(calc_activation(net, 'heaviside', threshold=0.1))

    # Task5 The following arrays show the output produced by a convolutional
    # layer
    # to all 4 samples in a batch
    # X1 = [ 1    0.5  2  ]
    #      [-1   -0.5 -0.2]
    #      [ 0.1 -0.1  0  ]
    #
    # X2 = [1   -1    0.1]
    #      [0.5 -0.5 -0.1]
    #      [0.2 -0.2  0  ]
    #
    # X3 = [0.5 -0.5 -0.1]
    #      [0   -0.4  0  ]
    #      [0.5  0.5  0.2]
    #
    # X4 = [ 0.2  1   -0.2]
    #      [-1   -0.4 -0.1]
    #      [ 0.1  0    0.1]
    # Calculate the corresponding outputs produced after the application of
    # batch normalisation, assuming the following parameter values β = 0,
    # γ = 1, and ε = 0.1 which are the same for all neurons
    print('batch normalization')
    X1 = np.array([[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]])
    X2 = np.array([[1, -1, 0.1], [0.5, -0.5, -0.1], [0.2, -0.2, 0]])
    X3 = np.array([[0.5, -0.5, -0.1], [0, -0.4, 0], [0.5, 0.5, 0.2]])
    X4 = np.array([[0.2, 1, -0.2], [-1, -0.6, -0.1], [0.1, 0, 0.1]])

    for output in batch_normalization([X1, X2, X3, X4], 0, 1, 0.1):
        print(str(output) + '\n')
