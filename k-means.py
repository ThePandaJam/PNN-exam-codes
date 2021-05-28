import numpy as np

# Initialisation - Parameters from question
k = 2
centers = {  # Initial clusters' centers
    0: [-1, 3],
    1: [5, 1],
}

x = np.array([  # Dataset
    [-1, 3],
    [1, 4],
    [0, 5],
    [4, -1],
    [3, 0],
    [5, 1]
])

try:
    assert(k == len(centers))
except AssertionError as e:
    e.args += ('The number of initialised clusers doesn\'t match k',
               len(centers), k)
    raise

print(
    f'The parameters are: k = {k} and the centers are {centers}')

# -----------------------------------------------------------------------------------


def distance(feautre, centers, method='euclidian'):
    if method == 'euclidian':
        return [np.linalg.norm(feature-centers[center])  # euclidian norm
                for center in centers]
    if method == 'manhatan':
        return [np.linalg.norm(feature-centers[center], 1)  # manhatan dist
                for center in centers]


previous = {}
for i in range(5):  # Max number of iterations (I don't think we'd be expected to perform more than 5 iterations)
    print(f'\n Iteration number {i+1}: \n')
    classes = {}  # Dict to hold a list of data points that are closest to the cluster number
    for j in range(k):
        classes[j] = []  # Instantiate empty list at every iteration
    for feature in x:  # For each data point do:
        # Compute the distance to each cluster (options: euclidian or manhatan)
        distances = distance(feature, centers, 'euclidian')

        classification = distances.index(
            min(distances))  # Find the lowest distance
        # Assign the datapoint to that cluster
        classes[classification].append(feature)

    # Print to which cluster each data point belongs to
    print(f'The classification is: {classes}')

    previous = centers.copy()  # Copy the cluster centers dict
    print(f'The previous centers are:{previous}')
    for classification in classes:
        # Compute the new cluster average by taking the mean of all the data point assigned to that cluster
        centers[classification] = np.average(classes[classification], axis=0)
    print(f'The new centers are:{centers}')

    opti = True
    for center in centers:
        prev = previous[center]  # Previous cluster
        curr = centers[center]  # Update cluster
        # termination criteria #TODO: IMPORTANT: This can be changed in the exam question!!
        if np.sum(curr-prev) != 0:
            opti = False
    if opti:
        break

print('\n The algorithm converged!')
