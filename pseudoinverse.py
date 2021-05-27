import numpy as np

# ------------------------------------------------------------------------------------
# Given parameters:
b = np.ones((6, 1))

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

print("Vectors used in Pseudoinverse operation to calculate parameters of linear discriminant function:\n {}\n".format(Norm_Y))

# ------------------------------------------------------------------------------------
# Initialising Y Matrix:
Y_matrix = []

# Adding each normalised sample in dataset to Y Matrix:
for i in range(len(Norm_Y)):
    Y_matrix.append(Norm_Y[i])
Y_matrix = np.array(Y_matrix)
print("y Matrix being used:\n {}\n".format(Y_matrix))

# Calculating pseudo-inverse of Y Matrix:
pseudo_inv_matrix = np.linalg.pinv(Y_matrix)
print("Pseudo-inverse Matrix is:\n {}\n".format(pseudo_inv_matrix))

# Multiplying Pseudo-inverse matrix by given margin vector in question:
a = np.matmul(pseudo_inv_matrix, b)
print("a is equal to:\n {}\n".format(a))

correct_classification = 0

# Checking if classifications are correct:

for sample in Norm_Y:
    ay = np.dot(sample, a)
    print("\ng(x) for sample {} is {}".format(sample, ay))

    # Sample is correctly classified if ay is positive:    
    if ay > 0:
        print("Sample has been correctly classified.")
        correct_classification += 1

if correct_classification == len(Norm_Y):
    print("\nAll samples are classified correctly which means that discriminant function parameters are correct.")

else:
    print("\nSome samples are misclassified.")

# ------------------------------------------------------------------------------------
