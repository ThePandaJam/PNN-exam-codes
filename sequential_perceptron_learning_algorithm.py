import numpy as np

# ------------------------------------------------------------------------------------
# Given Parameters in Question:

a = [1, 0, 0]         # Initial a
n = 1                 # Learning Rate

X = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
Y = [1, 1, 1, -1, -1, -1]

# ------------------------------------------------------------------------------------
# Applying Sample Normalisation:
Norm_Y = []

for x, y in zip(X, Y):
    
    # If the sample belongs to the class with label 2 or -1 (Check dataset in question to see how formatted):
    if y == -1 or y == 2:
        x = [i * -1 for i in x]
        x.insert(0, -1)
        Norm_Y.append(x)
    else:
        x.insert(0, 1)
        Norm_Y.append(x)

print("Vectors used in Sequential Perceptron Learning Algorithm:\n {}\n".format(Norm_Y))


# ------------------------------------------------------------------------------------
# Sequential Perceptron Learning Algorithm:

epoch = 1

while True:

    updating_samples = []
    print("Epoch {}".format(epoch))

    # Keeping track of how many samples are correctly classified. If this variable reaches 
    # the value that is equal to the size of the dataset (len), than we know that learning 
    # has converged:
    correctly_classified_counter = 0

    # Going through all of the samples in the dataset one-by-one:
    for i in range(len(Norm_Y)):
    
        # This chooses which weight to use for an iteration. If first iteration, uses given starting weight 
        # as described in question:
        a_prev = a
        print("The value of a used is {}".format(a_prev))
        
        # Selecting sample to use:
        y_input = Norm_Y[i]
        print("y Value used for this iteration is: {}".format(y_input))

        # Equation -> g(x) = a^{t}y
        ay = np.dot(a, y_input)
        print("The value of a^t*y for this iteration is: {}".format(ay))


        # Checking if the sample is misclassified or not:
        
        # If sample is misclassified:
        if ay <= 0:

            print("This sample is misclassified. This sample will be used in update.\n")
            updating_samples.append(np.array(a))
            updating_samples.append(np.array(y_input))

            # Calculating new value of a using update rule for Sequential Perceptron Learning Algorithm:
            a_update_val = n * sum(updating_samples)

            a = a_update_val
            print("\nNew Value of a^t is: {}.\n".format(a))
        
        # If the sample is correctly classified:
        else: 
            print("This sample is classified correctly.\n")
            correctly_classified_counter += 1
            pass
            
        # Reset sample to add for update to occur:
        updating_samples = []

    # If Block to check whether learning has converged. If we have gone through all the data without needing 
    # to update the parameters, we can conclude that learning has converged.
    if correctly_classified_counter == len(Norm_Y):
        print("\nLearning has converged.")
        print("Required parameters of a are: {}.".format(a))
        break
 
    epoch += 1