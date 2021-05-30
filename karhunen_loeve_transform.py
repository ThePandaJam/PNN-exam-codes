import numpy as np
from scipy.linalg import svd


def transform(ip, n_components, data_to_project=[]):
    ip = np.array(ip)
    N, D = ip.shape
    ip_mean = np.mean(ip, axis=0)
    ip_prime = ip - ip_mean
    ip_prime = ip_prime.T
    C = (ip_prime) @ (ip_prime.T)
    C = C / N
    V, D, VT = svd(C)
    p_input = VT @ (ip_prime)
    print("-"*100)
    print("Diagonal matrix [EIGEN VALUES]")
    print(np.round(np.diag(D), 3))
    print()
    print("EIGEN VECTOR[TRANSPOSED]")
    print(VT)
    print()
    print(
        "PROJECTION OF INPUT [READ  COLUMNS FROM TOP, TILL N_COMPONENTS ROWS]")
    print(np.round(p_input, 3))
    if len(data_to_project) > 0:
        print("PROJECTION OF TEST DATA")
        p_given_data = VT @ data_to_project.T
        print("READ COLUMN WISE")
        print(p_given_data)


# ip = [[1, 2, 1], [2, 3, 1], [3, 5, 1], [2, 2, 1]]
ip = [[0, 1], [3, 5], [5, 4], [5, 6], [8, 7], [9, 7]]
data_to_project = []
# INPUT A DATA COLUMN WISE X = [X1,X2,X3] -> IT SHOULD BE A COLUMN VECTOR
# ip = [[4, 2, 2], [0, -2, 2], [2, 4, 2], [-2, 0, 2]]
n_components = 2
# data_to_project = np.array([[3, -2, 5]])
transform(ip, n_components=n_components, data_to_project=data_to_project)
