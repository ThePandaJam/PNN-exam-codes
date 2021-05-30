import numpy as np
from collections import Counter
# THIS PROGRAM USES AUGMENTED NOTATION
# PREDICTION RULE - if multiple max value then choose max value with lowest index


def main(ip, dis_functions, targets, learning_rate=1):
    ip = np.array(ip)
    dis_functions = np.array(dis_functions)
    for _iter in range(2):
        for idx, x in enumerate(ip):
            x = np.insert(x, 0, 1)
            ground_truth = targets[idx]
            tmp = []
            for df in dis_functions:
                p = df @ x
                print(p, end=" ")
                tmp.append(p)
            tmp = np.array(tmp)
            max_value = np.max(tmp)
            counter_flag = False
            if Counter(tmp).most_common(1)[0][1] > 1:
                counter_flag = True
                # choose min index of max_value as prediction
                pred = np.argmax(tmp) + 1
                # choose max index of max_value as prediction
                # pred = len(tmp) - np.argmax(tmp[::-1])
            else:
                pred = np.argmax(tmp) + 1
            print(pred, ground_truth, end=" ")
            if (pred != ground_truth) or counter_flag:
                print()
                print("WEIGHT IS UPDATING: ")

                dis_functions[ground_truth -
                              1] = dis_functions[ground_truth-1] + (learning_rate * x)
                dis_functions[pred-1] = dis_functions[pred-1] - \
                    (learning_rate * x)
                print(dis_functions)
                print("-"*100)

            print()


# IP NOT AUGMENTATED

ip = [[0, 1, 0], [1, 0, 0], [0.5, 0.5, 0.25], [1, 1, 1], [0, 0, 0]]
targets = [1, 1, 2, 2, 3]
# AUGMENTED
dis_functions = [[1, 0.5, 0.5, -0.75], [-1, 2, 2, 1], [2, -1, -1, 1]]

# ip = [[1, 1], [2, 0], [0, 2], [-1, 1], [-1, -1]]
# dis_functions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
# targets = [1, 1, 2, 2, 3]
main(ip, dis_functions, targets)
