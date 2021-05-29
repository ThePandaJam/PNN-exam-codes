import numpy as np


def _svm(X, y, support_vectors, support_vector_class):
    X = np.array(X)
    y = np.array(y)

    print("-"*100)
    # support_vectors = np.array([[3, 1], [3, -1], [1, 0]])
    # support_vector_class = np.array([1, 1, -1])
    w = []
    for idx in range(len(support_vectors)):
        w.append(support_vectors[idx] * support_vector_class[idx])
    w = np.array(w)
    eq_arr = []
    for idx, sv in enumerate(support_vectors):
        tmp = ((w @ sv) * support_vector_class[idx])
        tmp = np.append(tmp, [support_vector_class[idx]])
        eq_arr.append(tmp)
    eq_arr.append(np.append(support_vector_class, [0]))
    rhs_arr = [1] * len(support_vector_class)
    rhs_arr.extend([0])
    rhs_arr = np.array(rhs_arr)
    try:
        ans = rhs_arr @ np.linalg.inv(eq_arr)
    except:
        print("Unabelt to do inverse, taking pseudo inverse")
        ans = rhs_arr @ np.linalg.pinv(eq_arr)
    print("lambda and w_0 values are ", ans)
    final_weight = []
    for idx in range(w.shape[0]):
        final_weight.append(w[idx] * ans[idx])
    final_weight = np.array(final_weight)
    final_weight = np.sum(final_weight, axis=0)
    print("Weights: ")
    print(final_weight)
    print("Margin: ")
    print(2/np.linalg.norm(final_weight))
    print("-"*100)


# REPLACE X AND Y ACCORDING TO THE QUESTION
# REPLACE "support_vectors" and "support_vector_class"
# X = [[3, 1], [3, -1], [7, 1], [8, 0], [1, 0], [0, 1], [-1, 0], [-2, 0]]
# y = [1, 1, 1, 1, -1, -1, -1, -1]
X = [[3, 3], [9, 7], [5, 5], [7, 9], [-1, -2], [-1, -4], [-3, -4], [-3, -2]]
y = [1, 1, 1, 1, -1, -1, -1, -1]
support_vectors = np.array(
    [[3, 1], [3, -1], [1, 0]])
support_vector_class = np.array([1, 1, -1])
# support_vectors = np.array([[3, 3], [-1, -2]])
# support_vector_class = np.array([1, -1])
# X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
# y = np.array([1, 1, -1, -1])
# support_vectors = X
# support_vector_class = y
_svm(X, y, support_vectors=support_vectors,
     support_vector_class=support_vector_class)
