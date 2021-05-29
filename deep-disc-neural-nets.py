import numpy as np

# Deep Discriminative Neural Networks [week 5]
from scipy import signal

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
    return np.heaviside(x - threshold, 0.5)


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


# Below code through the good offices of Hankun Wang
def strided_len(x_len, H_len, stride):
    return np.ceil((x_len - H_len + 1) / stride).astype(int)


def H_dilated_len(H_len, dilation):
    return (H_len - 1) * (dilation - 1) + H_len


def dilate_H(H, dilation):
    H_rows, H_cols = H[0].shape
    H_dilated = np.zeros((H.shape[0], H_dilated_len(H_rows, dilation),
                          H_dilated_len(H_cols, dilation)))
    H_dilated[:, ::dilation, ::dilation] = H
    return H_dilated


def apply_mask(X, H, padding=0, stride=1, dilation=1):
    # x and H can have multiple channels in the 0th dimension
    if padding > 0:
        X = np.pad(X, pad_width=padding, mode='constant')[1:-1]

    if dilation > 1:
        H = dilate_H(H, dilation)

    H_rows, H_cols = H[0].shape
    x_rows, x_cols = X[0].shape

    fm_rows, fm_cols = strided_len(x_rows, H_rows, stride), strided_len(x_cols,
                                                                        H_cols,
                                                                        stride)
    feature_map = np.empty((fm_rows, fm_cols))
    for xf in range(fm_rows):
        for yf in range(fm_cols):
            xi, yi = xf * stride, yf * stride
            receptive_region = X[:, xi: xi + H_rows, yi: yi + H_cols]
            feature_map[xf, yf] = np.sum(H * receptive_region)
    return feature_map


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

    # Task6 The following arrays show the feature maps that provide the input
    # to a convolutional layer of a CNN
    # X1 =   [ 0.2,  1. ,  0. ]
    #        [-1. ,  0. , -0.1]
    #        [ 0.1,  0. ,  0.1]
    #
    # X2 =   [ 1. ,  0.5,  0.2]
    #        [-1. , -0.5, -0.2]
    #        [ 0.1, -0.1,  0. ]
    #
    # h1 =  [ 1. , -0.1]
    #       [ 1. , -0.1]
    #
    # h2 =   [ 0.5,  0.5]
    #        [-0.5, -0.5]
    #
    X1 = [[0.2, 1., 0.],
          [-1., 0., -0.1],
          [0.1, 0., 0.1]]

    X2 = [[1., 0.5, 0.2],
          [-1., -0.5, -0.2],
          [0.1, -0.1, 0.]]

    X = np.array([X1, X2])

    h1 = [[1., -0.1],
          [1., -0.1]]

    h2 = [[0.5, 0.5],
          [-0.5, -0.5]]

    H = np.array([h1, h2])
    # Calculate the output produced by mask H when using:
    print('padding=0 and stride=1')
    print(apply_mask(X, H))

    print('padding=1 and stride=1')
    print(apply_mask(X, H, padding=1))

    print('padding=1 and stride=2')
    print(apply_mask(X, H, padding=1, stride=2))

    print('padding=0 and stride=1 dilation=2')
    print(apply_mask(X, H, padding=0, stride=1, dilation=2))
