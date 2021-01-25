import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from knn import KNN

if __name__ == '__main__':
    # Load hc_dataset dataset
    hc_dataset = pd.read_csv('HC_Body_Temperature.txt', sep="\s+", header=None, names=["x", "label", "y"], dtype=float)
    hc_dataset.loc[hc_dataset['label'] > 1, 'label'] = -1
    # print(hc_dataset)

    iterations = 1
    # ------------------------ Adaboost for hc_dataset ------------------------
    # adaboost_hc = Adaboost()
    # adaboost_hc.find_all_possible_lines(np.array(hc_dataset['x']), np.array(hc_dataset['y']))

    clf = KNN(k={1, 3, 5, 7, 9})
    hc_errors = pd.DataFrame({'emp_err': [0, 0, 0, 0, 0, 0, 0, 0], 'true_err': [0, 0, 0, 0, 0, 0, 0, 0]})
    for _ in range(iterations):
        x_train, x_test, y_train, y_test = train_test_split(hc_dataset[['x', 'y']], hc_dataset['label'], test_size=0.5)

        # print(np.array(x_train))
        p1 = np.array(x_train.iloc[0][['x', 'y']])
        p2 = np.array(x_train.iloc[1][['x', 'y']])
        # print(p1)
        # print(p2)
        # print(np.sqrt(np.sum((p1 - p2) ** 2)))

        clf.fit(np.array(x_train[['x', 'y']]), np.array(y_train))
        predictions = clf.predict(np.array(x_test[['x', 'y']]))

        # emp_errs = adaboost_hc.calc_errors(np.array(x_train['x']), np.array(x_train['y']), np.array(y_train))
        # true_errs = adaboost_hc.calc_errors(np.array(x_test['x']), np.array(x_test['y']), np.array(y_test))
        #
        # hc_errors['emp_err'] += emp_errs
        # hc_errors['true_err'] += true_errs

    # hc_errors /= iterations
    # print('-------------- HC dataset --------------')
    # print(hc_errors)
