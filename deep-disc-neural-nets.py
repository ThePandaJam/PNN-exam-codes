import numpy as np

# Deep Discriminative Neural Networks [week 5]

ERR_MSG = 'Please use proper activation function name. (relu,' \
          'lrelu,tanh,heaviside'


# The following array show the output produced by a mask in a convolutional
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

# Returns values after applying activation function.
# Params: net - a 2D numpy ndarray, activation - activation function, args to
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


def lrelu(x, a=0.1,**kwargs):
    return x if x >= 0 else a * x


def tanh(x, **kwargs):
    return np.tanh(x)


def heaviside(x, threshold=0.1,**kwargs):
    activation = x - threshold
    if activation > 0:
        return 1
    elif activation == 0.0:
        return 0.5
    else:
        return 0


if __name__ == '__main__':
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
