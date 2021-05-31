# Multilayer Perceptrons and Backpropagation [week 4]

import numpy as np
from deep_disc_neural_nets import relu, lrelu, tanh, heaviside

# returns dot product of each row with the features
# feature go in augmented one per row
def eval_discriminants(weights, features):
    return weights @ np.transpose(features)

# takes a function pointer from the deep_disc_neural_nets class
# kwargs is additional arg for activation function for example threshold
def eval_layer(weights, features, ptr_activation_func, **kwargs):
    return ptr_activation_func(eval_discriminants(weights, features), **kwargs)

## can chain these layers to evaluate a full network rapiddddddd
if __name__ == '__main__':
    features = np.array([[1, 1, 0, 1, 0], [1, 0, 1, 0, 1]])

    weights = np.array([[4.8432, -0.7057, 1.9061, 2.6605, -1.1359],
                       [0.3973, 0.4900, 1.9324, -0.4269, -5.1570],
                       [2.1761, 0.9438, -5.4160, -0.3431, -0.2931]])

    print(eval_layer(weights, features, heaviside, threshold=0.9))




