import numpy as np


def fishers(ip, weights, classes):
    ip = np.array(ip)
    N, D = ip.shape
    weights = np.array(weights)
    m1 = []
    m2 = []
    for idx in range(N):
        if classes[idx] == 1:
            m1.append(ip[idx])
        else:
            m2.append(ip[idx])
    m1 = np.mean(m1, axis=0)
    m2 = np.mean(m2, axis=0)

    # between cluster distance
    sb = []
    sw = []
    for w in (weights):
        d = (w @ (m1-m2)) ** 2
        sb.append(d)
    # calculate within cluster distance
    sw = []
    for w in weights:
        running_sw = 0
        for idx in range(len(ip)):
            if classes[idx] == 1:
                running_sw += (w.T @ (ip[idx] - m1)) ** 2

            elif classes[idx] == 2:
                running_sw += (w.T @ (ip[idx] - m2)) ** 2
        sw.append(running_sw)
        # print(running_sw)
    print("SB: ")
    print(sb)
    print("SW: ")
    print(sw)
    cost = []
    for _sb, _sw in zip(sb, sw):
        cost.append(_sb/_sw)
    print("Cost: ")
    print(cost)

    print("-"*100)
    print(f"{weights[np.argmax(cost)]} has high PROJECTION COST")


ip = [[1, 2], [2, 1], [3, 3], [6, 5], [7, 8]]
classes = [1, 1, 1, 2, 2]
weights = [[-1, 5], [2, -3]]
fishers(ip, weights, classes)
