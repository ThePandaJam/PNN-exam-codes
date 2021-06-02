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


# Below 4 functions for Q6 to Q8 is provided
# through the good offices of Hankun Wang
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

    fm_rows = strided_len(x_rows, H_rows, stride)
    fm_cols = strided_len(x_cols, H_cols, stride)

    feature_map = np.empty((fm_rows, fm_cols))
    for xf in range(fm_rows):
        for yf in range(fm_cols):
            xi, yi = xf * stride, yf * stride
            receptive_region = X[:, xi: xi + H_rows, yi: yi + H_cols]
            feature_map[xf, yf] = np.sum(H * receptive_region)
    return feature_map


def pooling(x, pooling_function, region=(2, 2), stride=1):
    x_rows, x_cols = x[0].shape
    pool_rows = strided_len(x_rows, region[0], stride)
    pool_cols = strided_len(x_cols, region[1], stride)

    pool = np.empty((x.shape[0], pool_rows, pool_cols))
    for xp in range(pool_rows):
        for yp in range(pool_cols):
            xi = xp * stride
            yi = yp * stride
            pooling_region = x[:, xi: xi + region[0], yi: yi + region[1]]
            pool[:, xp, yp] = pooling_function(pooling_region)
    return pool


# Returns output of the height, width and output size based on input shape,
# mask shape, number of masks, stride and padding. When defining shape for
# input or mask specify it as (height,width, n_channels)
def calc_output_dim(input_shape, mask_shape, n_masks, stride, padding):
    output_h = int(calc_dim(input_shape[0], mask_shape[0], padding, stride))
    output_w = int(calc_dim(input_shape[1], mask_shape[1], padding, stride))

    return output_h, output_w, output_h * output_w * n_masks


# Returns output size for specific dimension, like height or width in case of
# 2D arrays, based on input dimension, mask dimension, padding and stride
def calc_dim(input_dim, mask_dim, padding, stride):
    return 1 + ((input_dim - mask_dim + 2 * padding) / stride)


if __name__ == '__main__':

 print(calc_output_dim((200,200,3), (5,5,3), 40, 1 , 0))
