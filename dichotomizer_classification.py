import numpy as np

# Based on Question 1 in Tutorial 2:

def makeClassification(weight, intercept, feature_vector):
    
    weight = np.array(weight)
    feature_vector = np.array(feature_vector)

    wx = np.dot(weight, feature_vector)

    # Finding g(x):
    gx = wx + intercept

    if gx > 0:
        print("Feature Vector {} is in Class 1.".format(feature_vector))
    else:
        print("Feature Vector {} is in Class 2.".format(feature_vector))


if __name__ == '__main__':

    # determine the class of the following feature vectors:
    # Change these parameters to fit the question at hand.

    w = [2, 1]
    w_0 = -5


    x_1 = [1, 1]
    x_2 = [2, 2]
    x_3 = [3, 3]

    makeClassification(w, w_0, x_1)
    makeClassification(w, w_0, x_2)
    makeClassification(w, w_0, x_3)
