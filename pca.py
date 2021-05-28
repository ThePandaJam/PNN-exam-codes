import numpy as np
from scipy.linalg import svd


def _PCA(ip, n_components, data_to_project=None):
    ip = np.array(ip)
    ip_mean = np.mean(ip, axis=1)
    ip_prime = ip - np.vstack(ip_mean)
    C = (ip_prime @ ip_prime.T) / ip.shape[1]
    V, D, VT = svd(C)
    ans = VT @ ip_prime
    print("-"*100)
    print("READ THE ROWS FROM THE TOP")
    print(ans[:n_components])
    print("-"*100)
    if data_to_project:
        data_to_project = np.array(data_to_project)
        print("-"*100)
        print(f"PROJECTION OF {data_to_project}")
        print((VT@data_to_project)[:n_components])
        print("-"*100)


# REPLACE ACCORDING TO THE QUESTION
ip = [[4, 0, 2, -2], [2, -2, 4, 0], [2, 2, 2, 2]]
n_components = 2
data_to_project = [3, -2, 5]
_PCA(ip=ip, n_components=n_components, data_to_project=data_to_project)
