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
    print(discriminants)
    return np.transpose(ptr_activation_func(discriminants, **kwargs))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## can chain these layers to evaluate a full network rapiddddddd
## if node layers are not conencted fudge the weights as 0!
if __name__ == '__main__':
    features = np.array([[0.1, 0.9]])

    input_weights = np.array([[0.2, 0.5, 0],
                       [0, 0.3, -0.7]])

    hidden_weights = np.array([[-0.4, 0.8, 1.6]])
    hidden_layer = eval_layer(input_weights, features, tanh)
    print(hidden_layer)
    output_layer = eval_layer(hidden_weights, hidden_layer, tanh)
    print(np.round(output_layer, 4))


