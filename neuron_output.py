import numpy as np

# ------------------------------------------------------------------------------------

# Based on Question 2 in Tutorial 3:

def find_output(weight, threshold, input):
    
    summation = []

    for i in range(len(input)):
        summation.append(weight[i] * input[i])

    summation = np.sum(summation, 0) - threshold

    # Find output of neuron by applying heaviside function with given threshold:
    output = np.heaviside(summation, threshold)
    print("Output of neuron with input {} is {}.".format(input, output))


if __name__ == '__main__':
    
    # Update these parameters as necessary in the question:

    find_output(weight = [0.1, -5, 0.4], threshold = 0, input = [0.1, -0.5, 0.4])
    find_output(weight = [0.1, -5, 0.4], threshold = 0, input = [0.1, 0.5, 0.4])