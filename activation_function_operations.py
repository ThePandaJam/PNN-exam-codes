import numpy as np
import math

'''
Based on Question 4 in Tutorial 5:
The following array show the output produced by a mask in a convolutional layer of a CNN.

          [[1, 0.5, 0.2], 
 net_j =  [-1, -0.5, -0.2], 
          [0.1, -0.1, 0]]

Calculate the values produced by the application of the following activation functions:
'''

def main(net_j, activation_function, a = 0.1, threshold = 0.1, heaviside_0 = 0.5):

    new_array = []

    if activation_function == 'ReLU':

        for row in net_j:
            temp_array = []
            for i in row:
                
                # threshold:
                if i >= 0:
                    temp_array.append(i)
                else:
                    temp_array.append(0)
            new_array.append(temp_array)
        
        print(new_array)

    elif activation_function == 'LReLU':

        for row in net_j:
            temp_array = []
            for i in row:
                
                # threshold:
                if i >= 0:
                    temp_array.append(i)
                else:
                    temp_array.append(round(a * i, 2))
            new_array.append(temp_array)
        
        print(new_array)
    
    elif activation_function == 'tanh':

        for row in net_j:
            temp_array = []
            for i in row:

                # Using equation of tanh activation function:
                temp_array.append(round((math.e**i - math.e ** -i) / (math.e**i + math.e ** -i), 5)) 
            new_array.append(temp_array)
        
        print(new_array)

    elif activation_function == 'heaviside':

        for row in net_j:
            temp_array = []
            for i in row:
                
                # subtracts threshold away from each value:
                i = i - threshold

                # applies heaviside function to value:
                temp_array.append(np.heaviside(i, heaviside_0))

            new_array.append(temp_array)
        
        print(new_array)


if __name__ == '__main__':

    # Change net_j dependent on what mask is given in the question:
    
    net_j = [[1, 0.5, 0.2], [-1, -0.5, -0.2], [0.1, -0.1, 0]]

    # Change parameters of main based on the activation function asked for:

    main(net_j, activation_function = 'ReLU')
    main(net_j, activation_function = 'LReLU', a = 0.1)
    main(net_j, activation_function= 'tanh')
    main(net_j, activation_function= 'heaviside', threshold = 0.1, heaviside_0 = 0.5)