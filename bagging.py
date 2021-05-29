import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import copy
from collections import Counter
from sklearn.metrics import accuracy_score

# RELACE X AND Y ACCORDING TO QUESTION
# IT SHOULD BE NUMPY ARRAY
X, y = make_classification(
    n_samples=10000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)
df = pd.DataFrame(np.c_[X, y])
n_weak_classifier = 10
# Bagging = Random sampling with replacement
clf_arr = [None] * n_weak_classifier
for index in range(n_weak_classifier):
    random_size = np.random.randint(low=0, high=df.shape[0])
    sample = df.sample(n=random_size)
    input_data = sample.iloc[:, :-1]
    target = sample.iloc[:, -1]
    clf = DecisionTreeClassifier()
    clf.fit(input_data, target)
    # print(input_data.shape)
    clf_arr[index] = copy.deepcopy(clf)
    print(
        f"Weak Classifier {index+1} Training Error: {1 - clf.score(input_data, target)}")

# Combine all weak classifier to get final result
# prediction on test data
# consider "X" and "y" is test data

pred = []
for index in range(X.shape[0]):
    test_data = X[index].reshape(1, -1)
    ground_truth = y[index]
    temp_pred_arr = []
    for clf in clf_arr:
        ans = clf.predict(test_data).reshape(-1)[0]
        temp_pred_arr.append(ans)
    final_pred = Counter(temp_pred_arr).most_common(1)[0][0]
    final_pred = int(final_pred)
    pred.append(final_pred)

print("Final Accuracy: ", accuracy_score(y, np.array(pred)))
