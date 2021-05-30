import numpy as np


def main(p, VT, x, _lambda):
    p = np.array(p)
    VT = np.array(VT)
    x = np.array(x)
    r_error = []
    for p in projections:
        val = x - VT @ p
        r_error.append(np.linalg.norm(val) + _lambda*np.count_nonzero(p))
    print("RECONSTRUCTION ERRORS: ")
    print(r_error)
    print(projections[np.argmin(r_error)], " for sparse coding")


# REPLACE ACCORDING TO THE QUESTION
# projections are nothing but y
# projections = [[1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, -1, 0]]
# VT = [[0.4, 0.55, 0.5, -0.1, -0.5, 0.9, 0.5, 0.45],
#       [-0.6, -0.45, -0.5, 0.9, -0.5, 0.1, 0.5, 0.55]]
# x = [[-0.05, -0.95]]
projections = [[1, 2, 0, -1], [0, 0.5, 1, 0]]
x = [[2, 3]]
VT = [[1, 1, 2, 1], [-4, 3, 2, -1]]
main(p=projections, VT=VT, x=x, _lambda=1)
