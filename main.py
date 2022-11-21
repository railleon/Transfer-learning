from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from sklearn.metrics import log_loss

from keras.models import Model
import keras.layers
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.regularizers import l2

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
from utils.layer_utils import AttentionLSTM

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Conv1D, BatchNormalization, MaxPool2D, MaxPooling1D
from keras.optimizers import Adam            # 优化器
import keras
import matplotlib.pyplot as plt
import numpy as np
import time
#from load_cifar10 import load_cifar10_data

def target_model_0(id):

    model = Sequential()
    model.add(Flatten(input_shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS)))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NB_CLASS, activation='tanh'))

    model.summary()

    # Loads ImageNet pre-trained data
    model.load_weights('trained-weights/BPNN(1225)3_fold_%d_weights.h5'%id)

    # Truncate and replace softmax layer for transfer learning
    #model.layers.pop()
    #model.outputs = [model.layers[-1].output]
    #model.layers[-1].outbound_nodes = []


    new_model=Sequential()

    for layer in model.layers[0:7]:  # 跳过最后一层
        new_model.add(layer)
        print(layer.name)
    #top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #top_model.add(Dense(64, activation='relu'))
    #new_model.add(Dense(32, activation='relu'))
    #new_model.add(Dropout(0.5))
    #new_model.add(Dense(16, activation='relu'))
    #new_model.add(Dropout(0.5))
    #new_model.add(Dense(NB_CLASS, activation='tanh'))
#

    new_model.summary()

    for layer in new_model.layers[0:5]:
        print (layer.name)
        layer.trainable = False



    return new_model



def outputtxt(accx,valaccx,lossx,vallossx,id):

    plt.figure(1)
    plt.plot(accx, "g--", label="Mean_squared_error of training data")
    plt.plot(valaccx, "g", label="Mean_squared_error of validation data")
    plt.plot(lossx, "r--", label="Loss of training data")
    plt.plot(vallossx, "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss_%d'%id)
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()

    plt.figure(1).savefig(r'recorded results\acc__'+num+'_%d.jpg'%id)

    plt.close('all')

    acc=np.array(accx)
    np.savetxt(r'recorded results\acc__'+num+'_%d.txt'%id, acc)

    acc=np.array(valaccx)
    np.savetxt(r'recorded results\val_acc__'+num+'_%d.txt'%id, acc)

    acc=np.array(lossx)
    np.savetxt(r'recorded results\loss__'+num+'_%d.txt'%id, acc)

    acc=np.array(vallossx)
    np.savetxt(r'recorded results\val_loss__'+num+'_%d.txt'%id, acc)

if __name__ == '__main__':

    DATASET_INDEX = 50

    epoch = 50
    lr=1e-3

    n='BPNN(1207)'
    num = n+'0'

    MAX_NB_VARIABLES, MAX_TIMESTEPS, NB_CLASS = 1, 4, 1  # Resolution of inputs

    target = True

    X_train_min = np.zeros((1, 4))
    X_train_range = np.zeros((1, 4))
    y_train_min = np.zeros((1, 1))
    y_train_range = np.zeros((1, 1))
    id=1




    model = target_model_0(id)

    if target:
        X_train_min = np.loadtxt('trained-weights\X_train_min_%d.txt'%id)
        X_train_range = np.loadtxt('trained-weights\X_train_range_%d.txt'%id)
        y_train_min = np.loadtxt('trained-weights\y_train_min_%d.txt'%id)
        y_train_range = np.loadtxt('trained-weights\y_train_range_%d.txt'%id)

    history, y_train_mean, y_train_std = train_model(model, DATASET_INDEX, dataset_prefix=num, dataset_fold_id=id,
                                                     epochs=epoch, batch_size=128, learning_rate=lr,
                                                     target=target, X_train_min=X_train_min,
                                                     X_train_range=X_train_range, y_train_min=y_train_min,
                                                     y_train_range=y_train_range)

    loss, accuracy = evaluate_model(model, DATASET_INDEX, dataset_prefix=num, dataset_fold_id=id, batch_size=512,
                                    predict=True,
                                    target=target, X_train_min=X_train_min, X_train_range=X_train_range,
                                    y_train_min=y_train_min,
                                    y_train_range=y_train_range
                                    )

    with open('001-accuracy_' + num + '_' + str(lr) + '-' + str(epoch) + '.txt', 'a') as file0:
        print(accuracy, file=file0)
    #        with open('time_' + num + '_%d.txt' % id, 'a') as file0:
    #            print('CNN1 Running time: %s Seconds' % (t2), file=file0)

    outputtxt(history.history['mean_squared_error'], history.history['val_mean_squared_error'],
              history.history['loss'], history.history['val_loss'], id)

