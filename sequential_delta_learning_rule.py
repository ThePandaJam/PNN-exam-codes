import numpy as np

# ------------------------------------------------------------------------------------
# Based on Question 5 in Tutorial 3:
# Input parameters as given in the question -> CHECK THESE BEFORE ANSWERING QUESTIONS

learning_rate = 1

theta = [-1 * (-0.5)] 
w_1 = [1]
w_2 = [1]

# Finds initial w vector to use:
w = [y for x in [theta, w_1, w_2] for y in x]

# Original dataset as given in question:
x_1 = [0, 0, 1, 1]
x_2 = [0, 1, 0, 1]

X = []
Y = [0, 0, 0, 1]

# Converts dataset into usable feature vectors:

for x_1, x_2 in zip(x_1, x_2):

    feature_vector = []
    feature_vector.append(x_1)
    feature_vector.append(x_2)
    feature_vector.insert(0, 1)
    
    X.append(feature_vector)
# ------------------------------------------------------------------------------------
# Sequential Delta Learning Rule:

# # loop for the number of epochs:
while True:

    # neuron_output (y_input= H(wx)):
    # this for loop is going through each instance of the dataset (row):
    correct_counter = 0

    for i in range(len(X)):
        w_prev = w
        #take the x_input row at hand:
        x_t = X[i]
        t = Y[i]

        #calculating y_input = H(wx) column:
        y = np.dot(w, x_t)
        hv = np.heaviside(y, 0.5)
        print("Result of Heaviside function H(wx) is {}.".format(hv))

        # If H(wx) == t then move onto next iteration:
        if hv == t:

            correct_counter += 1
            pass

        elif hv > t:

            # creating an empty list of zeros to insert update weight:
            update = np.zeros(len(x_t))
            # creating array for learning rate:
            learning_rate = np.full((len(x_t), 1), learning_rate)

            for j in range(len(x_t)):
                update[j] = w[j] - learning_rate[j]*x_t[j]
            w = update

            correct_counter = 0

        elif hv < t:
            # creating an empty list of zeros to insert update weight:
            update = np.zeros(len(x_t))
            # creating array for learning rate:
            learning_rate = np.full((len(x_t), 1), learning_rate)

            for j in range(len(x_t)):
                update[j] = w[j] + learning_rate[j]*x_t[j]
            w = update

            correct_counter = 0
        print("New value of w is {}\n".format(w))

    if correct_counter == len(X):
        print("Learning has converged, so required weights are: {}.".format(w))
        break


 


