import numpy as np


def fuzzy_k_means(data, K, b, iterations, weights):
    data = np.array(data)
    weights = np.array(weights)
    centers = [None] * K
    # centers = np.array(centers)
    data = np.array(data)
    for _iter in range(iterations):
        # calculate centers
        for idx, w in enumerate(weights):
            new_center = [None] * data.shape[1]
            for idy in range(data.shape[1]):
                new_center[idy] = w[idy]**b * data[:, idy]
            centers[idx] = (np.array(new_center).sum(axis=0)/(w**b).sum())
        # print("Centers", centers)
        new_weight_container = [None] * data.shape[1]
        for idx in range(data.shape[1]):
            new_weights = []
            for c in centers:
                val = np.linalg.norm(c - data[:, idx])

                val = (1 / val) ** (2/(b-1))
                new_weights.append(val)
            new_weights = np.array(new_weights)
            nw_sum = new_weights.sum()
            for idy, nw in enumerate(new_weights):
                new_weights[idy] = nw/nw_sum
            new_weight_container[idx] = new_weights

        new_weight_container = np.array(new_weight_container)
        for col_idx in range(new_weight_container.shape[1]):
            weights[col_idx] = new_weight_container[:, col_idx]
        print(F"CENTERS AFTER ITERATION {_iter+1} [READ COLUMN WISE]")
        print(np.array(centers).T)
        print("#"*100)
        print(f"WEIGHTS AFTER ITERATION {_iter + 1} [READ COLUMN WISE]")
        print(weights.T)
        print("-"*100)


# REPLACE A VALUES AS GIVEN IN THE QUESTION
data = [[-1, 1, 0, 4, 3, 5], [3, 4, 5, -1, 0, 1]]
fuzzy_k_means(data=data, K=2, weights=[
              [1, 0.5, 0.5, 0.5, 0.5, 0], [0, 0.5, 0.5, 0.5, 0.5, 1]], b=2, iterations=3)
