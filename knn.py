import numpy as np
import pandas as pd
from collections import Counter


def calc_err(y_true, y_pred):
    accuracy = np.sum(y_true != y_pred) / len(y_true)
    return accuracy


def distance(p1, p2):
    l_1 = np.sum(np.abs(p1 - p2))
    l_2 = np.sqrt(np.sum((p1 - p2) ** 2))
    l_inf = np.max(np.abs(p1 - p2))
    return [l_1, l_2, l_inf]


class KNN:

    def __init__(self, k={1, 3, 5, 7, 9}):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, y):
        data = []
        for k in self.k:
            y_pred = np.array([self._predict(x, k) for x in X])

            l_1_err = calc_err(y, y_pred[:, 0])
            l_2_err = calc_err(y, y_pred[:, 1])
            l_inf_err = calc_err(y, y_pred[:, 2])

            data.append([k, l_1_err, l_2_err, l_inf_err])

        predictions = pd.DataFrame(data, columns=['K_value', 'l_1', 'l_2', 'l_inf'])

        return predictions

    #       l1
    # k=1   ..
    #
    #
    #

    def _predict(self, x, k):
        # Compute distances between x and all examples in the training set
        distances = np.array([distance(x, x_train) for x_train in self.X_train])

        k_idx_l_1 = np.argsort(distances[:, 0])[:k]
        k_idx_l_2 = np.argsort(distances[:, 1])[:k]
        k_idx_l_inf = np.argsort(distances[:, 2])[:k]

        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels_l_1 = [self.y_train[i] for i in k_idx_l_1]
        k_neighbor_labels_l_2 = [self.y_train[i] for i in k_idx_l_2]
        k_neighbor_labels_l_inf = [self.y_train[i] for i in k_idx_l_inf]

        # return the most common class label
        most_common_l_1 = Counter(k_neighbor_labels_l_1).most_common(1)
        most_common_l_2 = Counter(k_neighbor_labels_l_2).most_common(1)
        most_common_l_inf = Counter(k_neighbor_labels_l_inf).most_common(1)

        return [most_common_l_1[0][0], most_common_l_2[0][0], most_common_l_inf[0][0]]
