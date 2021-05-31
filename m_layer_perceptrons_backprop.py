# Multilayer Perceptrons and Backpropagation [week 4]

import numpy as np
from deep_disc_neural_nets import relu, lrelu, tanh, heaviside

# returns dot product of each row with the features
# feature go in one per row
def eval_discriminants(weights, inputs):
    onez = np.ones((inputs.shape[0], 1))
    augmented_input = np.append(onez, inputs, axis=1)
    return weights @ np.transpose(augmented_input)

# takes a function pointer from the deep_disc_neural_nets class
# kwargs is additional arg for activation function for example threshold
def eval_layer(weights, inputs, ptr_activation_func, **kwargs):
    discriminants = eval_discriminants(weights, inputs)
    return np.transpose(ptr_activation_func(discriminants, **kwargs))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## can chain these layers to evaluate a full network rapiddddddd
if __name__ == '__main__':
    features = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])

    input_weights = np.array([[4.8432, -0.7057, 1.9061, 2.6605, -1.1359],
                       [0.3973, 0.4900, 1.9324, -0.4269, -5.1570],
                       [2.1761, 0.9438, -5.4160, -0.3431, -0.2931]])

    hidden_weights = np.array([[2.5230, -1.1444, 0.3115, -9.9812],
                               [2.6463, 0.0106, 11.547, 2.6479]])

    hidden_layer = eval_layer(input_weights, features, tanh)
    print(hidden_layer)
    output_layer = eval_layer(hidden_weights, hidden_layer, sigmoid)
    print(np.round(output_layer, 4))


