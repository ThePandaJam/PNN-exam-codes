import numpy as np

# This script is based off of Question 7 in Tutorial 3:

def activationOfOutput(weights, iterations, input, alpha, activations):

    prev_activations = activations
    iteration = 1

    for i in range(iterations):

        print("Iteration {}".format(iteration))
        
        # Following block deals with calculating first equation: e = x - W^{T}y
        wT = np.array(weights).T
        wTy = np.dot(wT, activations)
        print("value of wTy {}".format(wTy))

        eT = input - wTy
        print("eT: {}".format(eT))
        e = np.array(eT).reshape((3, 1))

        # The following lines deal with calculating the update: y <- y + \alpha*W*e
        We = np.dot(weights, e)
        We = [j for i in We for j in i]
        print("We: {} ".format(We))

        alphaWe = np.dot(alpha, We)

        # Doing the actual update using the second equation:
        y = activations + alphaWe
        print("Value of y: {}\n".format(y))

        activations = y

        iteration += 1

    print("\nAfter {} iterations, the activation of the output neurons is equal to {}".format(iterations, activations))

if __name__ == '__main__':
    
    # Change these parameters depending on what the question asks for:

    activationOfOutput(weights = [[1, 1, 0], [1, 1, 1]], iterations = 5, input = [1, 1, 0], alpha = 0.5, activations = [0, 0])