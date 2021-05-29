from typing import Counter
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import copy
from sklearn.metrics import accuracy_score


def draw_samples(df):
    random_size = np.random.randint(low=0, high=df.shape[0]//2)
    sample_df = df.sample(n=random_size, replace=False)
    return sample_df


def is_valid_sample(df):
    if len(np.unique(df.iloc[:, -1])) == 2:
        return True
    return False


X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0, random_state=0, shuffle=False)
df = pd.DataFrame(np.c_[X, y])
n_weak_classifiers = 10
# clf = AdaBoostClassifier(n_estimators=n_weak_classifiers, random_state=0)
# clf.fit(X, y)
# print("Score: ", clf.score(X, y))

# Boosting = sampling without replacement
clf_arr = [None] * n_weak_classifiers
prev_misclassified_samples = []
for index in range(n_weak_classifiers):
    sample_df = draw_samples(copy.deepcopy(df))
    while not is_valid_sample(copy.deepcopy(sample_df)):
        sample_df = draw_samples(copy.deepcopy(df))
    df = df.drop(sample_df.index, axis="rows")
    df = df.reset_index(drop=True)
    input_data = sample_df.iloc[:, :-1].values
    target = sample_df.iloc[:, -1].values
    if prev_misclassified_samples:
        if prev_misclassified_samples[-1]["feature_vectors"].shape[0] != 0:
            input_data = np.vstack(
                (input_data, prev_misclassified_samples[-1]["feature_vectors"]))

            target = list(target)
            target.extend(prev_misclassified_samples[-1]["targets"])
            target = np.array(target)

    clf = SVC()
    clf.fit(input_data, target)
    clf_arr[index] = copy.deepcopy(clf)
    pred_arr = []
    for _clf in clf_arr:
        if not _clf:
            break
        pred_arr.append(_clf.predict(input_data))
    f_pred_arr = [None] * input_data.shape[0]
    mis_classified_samples_index = []
    for idy in range(input_data.shape[0]):
        all_pred = []
        for pred in pred_arr:
            all_pred.append(pred[idy])
        if len(np.unique(all_pred)) != 1:
            mis_classified_samples_index.append(idy)
    prev_misclassified_samples.append(
        {"feature_vectors": input_data[mis_classified_samples_index], "targets": target[mis_classified_samples_index]})


# Test data ensemble bootstrap classifer
pred = []
for ip in X:
    ip = ip.reshape(1, -1)
    tmp_arr = []
    for _clf in clf_arr:
        tmp_arr.append(_clf.predict(ip)[0])
    ans = Counter(tmp_arr).most_common(1)[0][0]
    ans = int(ans)
    pred.append(ans)

print("Accuracy Score: ", accuracy_score(y, pred))
print(np.mean(y == pred))
