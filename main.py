import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from knn import KNN

if __name__ == '__main__':
    # Load hc_dataset dataset
    hc_dataset = pd.read_csv('HC_Body_Temperature.txt', sep="\s+", header=None, names=["x", "label", "y"], dtype=float)
    hc_dataset.loc[hc_dataset['label'] > 1, 'label'] = -1

    iterations = 100
    clf = KNN(k={1, 3, 5, 7, 9})

    data = [[1, 0, 0, 0], [3, 0, 0, 0], [5, 0, 0, 0], [7, 0, 0, 0], [9, 0, 0, 0]]
    avg_true_errs = pd.DataFrame(data, columns=['K_value', 'l_1', 'l_2', 'l_inf'])
    avg_emp_errs = pd.DataFrame(data, columns=['K_value', 'l_1', 'l_2', 'l_inf'])
    for _ in range(iterations):
        x_train, x_test, y_train, y_test = train_test_split(hc_dataset[['x', 'y']], hc_dataset['label'], test_size=0.5)

        clf.fit(np.array(x_train[['x', 'y']]), np.array(y_train))
        predictions_test = clf.predict(np.array(x_test[['x', 'y']]), np.array(y_test))
        predictions_train = clf.predict(np.array(x_train[['x', 'y']]), np.array(y_train))

        avg_true_errs[['l_1', 'l_2', 'l_inf']] += predictions_test[['l_1', 'l_2', 'l_inf']]
        avg_emp_errs[['l_1', 'l_2', 'l_inf']] += predictions_train[['l_1', 'l_2', 'l_inf']]

    avg_true_errs[['l_1', 'l_2', 'l_inf']] /= iterations
    avg_emp_errs[['l_1', 'l_2', 'l_inf']] /= iterations

    print("----------- true_err -----------")
    print(avg_true_errs)
    print("----------- emp_err -----------")
    print(avg_emp_errs)
