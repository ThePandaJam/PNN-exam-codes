import numpy as np

# ------------------------------------------------------------------------------------
# Given Parameters in Question:
a = [-25, 6, 3]         # Initial a
n = 1                   # Learning Rate

# Given Dataset:
X = [[1, 5], [2, 5], [4, 1], [5, 1]]
Y = [1, 1, 2, 2]


# ------------------------------------------------------------------------------------
# Applying Sample Normalisation:
Norm_Y = []

for x, y in zip(X, Y):
    
    # If the sample belongs to the class with label 2 or -1 (Check dataset in question to see how formatted):
    if y == 2 or y == -1:
        x = [i * -1 for i in x]
        x.insert(0, -1)
        Norm_Y.append(x)
    else:
        x.insert(0, 1)
        Norm_Y.append(x)

print("Vectors used in Batch Perceptron Learning Algorithm:\n {}\n".format(Norm_Y))

# ------------------------------------------------------------------------------------
# Batch Perceptron Learning Algorithm:

epoch = 1

while True:

    updating_samples = []
    print("Epoch {}".format(epoch))

    for count, i in enumerate(range(len(Norm_Y))):
    
        # Knowing which value of a to use. If it is the first iteration, than use the given parameters in the 
        # question:
        a_prev = a
        print("The value of a used is {}".format(a_prev))
        y_input = Norm_Y[i]
        print("y Value used for this iteration is: {}".format(y_input))

        # Equation -> g(x) = a^{t}y
        ay = np.dot(a, y_input)
        print("The value of a^t*y for this iteration is: {}".format(ay))
        

        # Checking if the sample is misclassified or not:
        
        # If sample is misclassified:
        if ay <= 0:

            # If this is the first sample in the epoch, add the previous value of a to the list of samples used 
            # for the update to perform summation at the end of the epoch:
            if count == 0:
                print("This sample is misclassified. This sample will be used in update.\n")
                updating_samples.append(np.array(a))
                updating_samples.append(np.array(y_input))
            
            # If sample is misclassified and IS NOT the first sample in the epoch:
            else:
                print("This sample is misclassified. This sample will be used in update.\n")
                updating_samples.append(np.array(y_input))
        
        # If sample is classified correctly:
        else: 

            # If first sample in the epoch, append the previous value of a to the updating samples list:
            if count == 0:
                updating_samples.append(np.array(a))
                print("This sample is classified correctly.\n")
            else:
                print("This sample is classified correctly.\n")

    # Calculating new value of a after having gone through all of the samples in the dataset since it is Batch Learning.
    a_update_val = n * sum(updating_samples)

    # If Block to check whether learning has converged. If we have gone through all the data without needing 
    # to update the parameters, we can conclude that learning has converged.
    if len(updating_samples) <= 1:
        print("\nLearning has converged.")
        print("Required parameters of a are: {}".format(a))
        break

    # Updating a using our new value of a:
    a = a_update_val
    print("\nNew Value of a^t is: {}.\n".format(a))

    epoch += 1




