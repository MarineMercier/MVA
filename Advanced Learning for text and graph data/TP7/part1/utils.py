"""
Learning on Sets - ALTEGRAD - Jan 2022
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    # your code here #
    ##################

    X_train= np.zeros((n_train, max_train_card))
    y_train = np.zeros(n_train)
    for i in range(n_train):
        card = np.random.randint(1, 11)
        X_train[i, -card:] = np.random.randint(1, 11, size=card)
        y_train[i] = np.sum(X_train[i,:])

    print("The first training sample is :", X_train[0,:])
    print("The first label is", y_train[0])

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    # your code here #
    ##################
    n_test = 200000
    min_test_card = 5
    max_test_card = 101
    step_test_cardinality = 5
    cards=range(min_test_card, max_test_card, step_test_cardinality)
    n_samples_card = n_test//len(cards)

    X_test = list()
    y_test = list()

    for card in cards:
        X_test.append(np.random.randint(1, 11, size=(n_samples_card, card)))
        y_test.append(np.sum(X_test[-1], axis=1))

    return X_test, y_test