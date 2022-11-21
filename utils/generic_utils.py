import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pylab as plt

mpl.style.use('seaborn-paper')

from utils.constants import TRAIN_FILES, TEST_FILES, MAX_NB_VARIABLES, NB_CLASSES_LIST


def load_dataset_at(index, fold_index=None, normalize_timeseries=False, verbose=True,target=False, X_trainmin=None, X_trainrange=None,
                                                                                                     y_trainmin=None,
                                                                                                     y_trainrange=None) -> (np.array, np.array):
    if verbose: print("Loading train / test dataset : ", TRAIN_FILES[index], TEST_FILES[index])

    if fold_index is None:
        x_train_path = TRAIN_FILES[index] + "X_train.npy"
        y_train_path = TRAIN_FILES[index] + "y_train.npy"
        x_test_path = TEST_FILES[index] + "X_test.npy"
        y_test_path = TEST_FILES[index] + "y_test.npy"
    else:
        x_train_path = TRAIN_FILES[index] + "X_train.npy"
        y_train_path = TRAIN_FILES[index] + "y_train_%d.npy" % fold_index
        x_test_path = TEST_FILES[index] + "X_test.npy"
        y_test_path = TEST_FILES[index] + "y_test_%d.npy" % fold_index

    print(x_train_path,'x_train_path',
          x_train_path[1:],'x_train_path[1:]')

    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        print('os.path.exists(x_train_path)')
        print(x_train_path)

    elif os.path.exists(x_train_path[1:]):
        X_train = np.load(x_train_path[1:])
        y_train = np.load(y_train_path[1:])
        X_test = np.load(x_test_path[1:])
        y_test = np.load(y_test_path[1:])

        print('os.path.exists(x_train_path[1:])')
        print(x_train_path[1:])
    else:
        raise FileNotFoundError('File %s not found!' % (TRAIN_FILES[index]))

    is_timeseries = True


    [x,y,z]=X_train.shape



    if target== True:

        X_train = (X_train - X_trainmin) / (X_trainrange)

        y_train = (y_train - y_trainmin) / (y_trainrange)



    else:
        X_train_min = X_train.min(axis=0)
        X_train_range = X_train.max(axis=0) - X_train.min(axis=0)
        X_train = (X_train - X_train_min) / (X_train_range)
        y_train_min = y_train.min(axis=0)
        y_train_range = y_train.max(axis=0) - y_train.min(axis=0)
        y_train = (y_train - y_train_min) / (y_train_range)




    if verbose: print("Finished processing train dataset..")

    # extract labels Y and normalize to [0 - (MAX - 1)] range

    X_test = (X_test - X_trainmin) / (X_trainrange )

    y_test = (y_test - y_trainmin) / (y_trainrange)


    print('X_train.shape',X_train.shape)

    np.savetxt('data situation\X_train.txt', X_train.reshape(-1,4), fmt='%1.9f')
    np.savetxt('data situation\X_test.txt', X_test.reshape(-1,4), fmt='%1.9f')
    np.savetxt('data situation\y_train.txt', y_train.reshape(-1,1), fmt='%1.9f')
    np.savetxt('data situation\y_test.txt', y_test.reshape(-1,1), fmt='%1.9f')

    a=np.zeros((1,1))
    a[0,0]=30

    aa=y_trainmin




    np.savetxt('data situation\X_train_min.txt', X_trainmin.reshape(1,-1), fmt='%1.9f')
    np.savetxt('data situation\X_train_range.txt', X_trainrange.reshape(1,-1), fmt='%1.9f')
    np.savetxt('data situation\y_train_min.txt', y_trainmin.reshape(1,-1), fmt='%1.9f')

    np.savetxt('data situation\y_train_range.txt', y_trainrange.reshape(1,-1), fmt='%1.9f')


    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])

        print("Sequence length : ", X_train.shape[-1])

    return X_train, y_train, X_test, y_test, is_timeseries,  y_trainmin, y_trainrange


def calculate_dataset_metrics(X_train):
    max_nb_variables = X_train.shape[1]
    max_timesteps = X_train.shape[-1]

    return max_timesteps, max_nb_variables


def cutoff_choice(dataset_id, sequence_length):
    print("Original sequence length was :", sequence_length, "New sequence Length will be : ",
          MAX_NB_VARIABLES[dataset_id])
    choice = input('Options : \n'
                   '`pre` - cut the sequence from the beginning\n'
                   '`post`- cut the sequence from the end\n'
                   '`anything else` - stop execution\n'
                   'To automate choice: add flag `cutoff` = choice as above\n'
                   'Choice = ')

    choice = str(choice).lower()
    return choice


def cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length):
    assert MAX_NB_VARIABLES[dataset_id] < sequence_length, "If sequence is to be cut, max sequence" \
                                                                   "length must be less than original sequence length."
    cutoff = sequence_length - MAX_NB_VARIABLES[dataset_id]
    if choice == 'pre':
        if X_train is not None:
            X_train = X_train[:, :, cutoff:]
        if X_test is not None:
            X_test = X_test[:, :, cutoff:]
    else:
        if X_train is not None:
            X_train = X_train[:, :, :-cutoff]
        if X_test is not None:
            X_test = X_test[:, :, :-cutoff]
    print("New sequence length :", MAX_NB_VARIABLES[dataset_id])
    return X_train, X_test


if __name__ == "__main__":
    pass