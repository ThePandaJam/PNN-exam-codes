import numpy as np

# ------------------------------------------------------------------------------------
# Given Parameters in Question:
a = [1, 0, 0]           # Initial value of a
b = np.ones((6,1))      # Margin Vector
n = 0.1                 # Learning Rate
iterations = 12         # Iterations


# Given Dataset:

# Add dataset as given in question. This is assuming that Sample Normalisation has NOT been applied:
X = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
Y = [1, 1, 1, -1, -1, -1]


# ------------------------------------------------------------------------------------
# Applying Sample Normalisation:
Norm_Y = []

for x, y in zip(X, Y):
    
    # If the sample belongs to the class with label 2 or -1 (Check dataset in question to see how formatted):
    if y == -1 or y == 2:
        x = [i * -1 for i in x]
        x.insert(0, y)
        Norm_Y.append(x)
    else:
        x.insert(0, y)
        Norm_Y.append(x)

print("Vectors used in Sequential Widrow-Hoff Learning Algorithm:\n {}\n".format(Norm_Y))


# ------------------------------------------------------------------------------------
# Sequential Widrow-Hoff Learning Algorithm

# Epoch for-loop:
for o in range(int(iterations / len(Norm_Y))):

    # This for-loop goes through each sample one-by-one:
    for i in range(len(Norm_Y)):

        # Value of a to use. If first iteration, then uses parameters given in question:
        
        a_prev = a
        
        # Which sample to use:
        y_input = Norm_Y[i]
        print("Sample used for this iteration is: {}".format(y_input))

        # Equation -> g(x) = a^{t}y
        ay = np.dot(a, y_input)
        print("g(x) = {}".format(ay))

        # Calculating the values for update:
        update = np.zeros(len(y_input))
        for j in range(len(y_input)): 
            
            # Applying Update Rule of Sequential Widrow-Hoff Learning Algorithm:
            update[j] = n * (b[i] - ay) * y_input[j]

        # Adding update to a:
        a = np.add(a, update)
        print("New Value of a^t is: {}\n".format(a))

print("Gone through all of the iterations as asked for in question.")

