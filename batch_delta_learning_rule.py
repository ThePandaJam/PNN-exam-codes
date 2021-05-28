import numpy as np

# ------------------------------------------------------------------------------------
# Based on Question 3 in Tutorial 3:
# Input parameters as given in the question -> CHECK THESE BEFORE ANSWERING QUESTIONS

learning_rate = 1

theta = [-1 * (1.5)] 
w_1 = [2]


# Finds initial w vector to use:
w = [y for x in [theta, w_1] for y in x]

# Original dataset as given in question:
x_1 = [0, 1]


X = []
Y = [1, 0]

# Converts dataset into usable feature vectors:

for value in x_1:

    feature_vector = []
    feature_vector.append(value)
    feature_vector.insert(0, 1)
    
    X.append(feature_vector)

print(X)
# ------------------------------------------------------------------------------------
# Batch Delta Learning Rule:

epoch = 1
# # loop for the number of epochs:
while True:


    # this for loop is going through each instance of the dataset (row):
    correct_counter = 0
    update_tracker = []

    for i in range(len(X)):

        # Weight to use for this epoch. If first epoch than use given parameters in question:
        w_prev = w
        print("w value being used for Epoch {} is {}".format(epoch, w))
    
        x_t = X[i]
        t = Y[i]

        # Calculating y_input = H(wx) column:
        y = np.dot(w, x_t)
        # Output of neuron -> (y_input= H(wx)):
        hv = np.heaviside(y, 0.5)
        print("Result of Heaviside function H(wx) is {}.".format(hv))


        # Checking whether result of heaviside function is equal to our target value:
        if t == hv:
            correct_counter += 1
        

        # Performing update for each feature vector in dataset:
        t_min_y = t - hv

        # creating an empty list of zeros to insert update weight:
        update = np.zeros(len(x_t))
        # creating array for learning rate:
        learning_rate = np.full((len(x_t), 1), learning_rate)

        # Calculating update value for feature vector at hand:
        for j in range(len(x_t)):
            update[j] = learning_rate[j] * t_min_y * x_t[j]
        
        update_tracker.append(update)

    # Calculating total weight change of the epoch:
    total_weight_change = np.sum(update_tracker, 0)  
    print("Total Weight Change of this Epoch is {}".format(total_weight_change))

    # Adding total weight change to previous weight in order to perform update:
    w = w + total_weight_change  
    print("New value of w is {}\n".format(w))
    epoch += 1

    # Checking whether learning has converged:
    if correct_counter == len(X):
        print("\nLearning has converged, so required weights are: {}.".format(w))
        print("This is equivalent to having theta = {}, and w_1 = {}.".format(w[0]*-1, w[1]))
        break






 


