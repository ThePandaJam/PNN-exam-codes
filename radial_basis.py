import numpy as np

# --------------------------------INPUT--------------------------------#
X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
)

Y = np.array([0, 1, 1, 0])

C = np.array(
    [
        [0, 0],
        [1, 1],
    ]
)
# ----------------------------------------------------------------------#


# ---------------------------------TEST---------------------------------#
X_predict = np.array(
    [
        [0.5, -0.1],
        [-0.2, 1.2],
        [0.8, 0.3],
        [1.8, 0.6]
    ]
)
# ----------------------------------------------------------------------#


# --------------Calculate rho and sigma for radial basis---------------#
center_dist = []
for i in range(0, len(C) - 1):
    for j in range(i + 1, len(C)):
        print(i, j)
        center_dist.append(np.sqrt(np.sum((C[i, :] - C[j, :]) ** 2)))

rho_max = np.max(center_dist)
rho_avg = np.average(center_dist)
nH = len(C)
print("Rho max: ", rho_max)
print("Rho average: ", rho_avg)

sigma = rho_max / np.sqrt(2 * nH)       # Using rho-max
# sigma = 2 * rho_avg                   # Using rho-average
print("Sigma: ", sigma)


# -----------------------------CALCULATE OUTPUTS---------------------------#

def get_hidden_output(X, C, sigma):
    radial_basis_output = []
    for i in range(0, len(X)):
        hidden_node_outputs = []
        for j in range(0, len(C)):
            # Using Gaussian function
            hidden_node_outputs.append(np.exp(-np.sum((X[i] - C[j]) ** 2)) / (2 * sigma * sigma))
        radial_basis_output.append(hidden_node_outputs)

    print("Radial basis layer output: ")
    print(np.round(radial_basis_output, 2))
    return radial_basis_output

# Get output from hidden rbf layer
radial_basis_output = get_hidden_output(X, C, sigma)

# Add bias to hidden layer output
radial_basis_output = np.c_[radial_basis_output, np.ones(len(radial_basis_output))]
radial_basis_output_transposed = np.transpose(radial_basis_output)

# Least squares method to calculate weights
weights = np.dot(
    np.dot(
        np.linalg.inv(
            np.dot(radial_basis_output_transposed, radial_basis_output)
        ), radial_basis_output_transposed
    ), Y
)
print("Weights between hidden-output layer: ", np.round(weights, 2))

# Get output of final layer
calculated_output = np.dot(radial_basis_output, weights)
print("Output of network: ", np.round(calculated_output, 2))

# Apply basic sign function with 0.5 threshold or call any other activation function
final_output = np.where(calculated_output > 0.5, 1, 0)
print("Signed output of network: ", final_output)

# ---------------------------------------------------------------------------#


# ------------------------------PREDICT TEST OUTPUT---------------------------#

# Get output of hidden rbf layer
hidden_output = get_hidden_output(X_predict, C, sigma)

# Get output of final output layer
predicted_output = np.dot(np.c_[hidden_output, np.ones(len(hidden_output))], weights)
print("Output of test samples: ", np.round(predicted_output, 2))

test_output = np.where(predicted_output > 0.5, 1, 0)
print("Signed Output of test samples: ", test_output)