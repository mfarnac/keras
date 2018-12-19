from __future__ import print_function
import keras
import hLayers as h
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, Input, Conv1D
import copy
from keras import backend as K
import tensorflow as tf
'''
cd Desk*/ker*
python rectFinder.py
'''


# data_set_type = 'cifar'
# data_set_type = 'cifar100'
# data_set_type = 'mnist'
# data_set_type = 'mnist_noise'
data_set_type = 'fashion_mnist'
# data_set_type = 'fashion_mnist_noise'
noiseImagesNumber = 10000


def doubleConvMP(numCores, (width, height), myInput):
    conv1 = Conv2D(numCores, (width, height), data_format=channel_style, padding='valid', activation='relu')(myInput)
    conv1 = Conv2D(numCores, (3, 3), data_format=channel_style, padding='valid', activation='relu')(conv1)
    conv1 = Conv2D(numCores, (height, width), data_format=channel_style, padding='valid', activation='relu')(conv1)
    conv1 = MaxPooling2D((2, 2), data_format=channel_style, strides=(1, 1), padding='valid')(conv1)
    conv1 = Flatten()(conv1)
    return conv1


print('\n' + data_set_type)
num_classes = 2
first_epochs = 2
first_training_batch_sizes = [128, 256]
final_epochs = 3
final_training_batch_sizes = [128, 256, 512, 1024]
initial_verbosity = 1
final_verbosity = 1

insideCores = 2

myLr = 0.001

channel_style = 'channels_last'
my_shape = [28, 28, 1]
if data_set_type == 'cifar':
    (x_train, y_train_i), (x_test, y_test_i) = h.getHcifar10()
    channel_style = 'channels_first'
    my_shape = [3, 32, 32]
elif data_set_type == 'cifar100':
    (x_train, y_train_i), (x_test, y_test_i) = h.getHcifar100()
    channel_style = 'channels_first'
    my_shape = [3, 32, 32]
    testRange = 100
elif data_set_type == 'fashion_mnist':
    (x_train, y_train_i), (x_test, y_test_i) = h.getHFmnistConv()
elif data_set_type == 'fashion_mnist_noise':
    (x_train, y_train_i), (x_test, y_test_i) = h.getHFmnistConvNoise(noiseImagesNumber)
elif data_set_type == 'mnist_noise':
    (x_train, y_train_i), (x_test, y_test_i) = h.getHmnistConvNoise(noiseImagesNumber)
else:
    (x_train, y_train_i), (x_test, y_test_i) = h.getHmnistConv()


h.printTime()

my_scores = []
my_rects = []
input1 = Input(shape=my_shape)

zucchini = [6, 2, 3, 4, 0, 1, 5, 7, 8, 9]
testRange = len(zucchini)
print('\nlabels tested: '+str(testRange))

# rects = [[3, 3], [3, 7], [3, 11], [3, 15]]
rects = [[3, 3], [7, 3], [11, 3], [15, 3]]
# rects = [[2, 2], [1, 1], [1, 4], [4, 1], [2, 5], [5, 2]]
#  [[3, 7], [5, 7], [7, 7], [5, 5], [7, 5], [7, 3],
#         [3, 9], [9, 5], [7, 9], [5, 3], [5, 9],
#         [9, 3], [3, 5], [13, 7], [3, 3], [11, 7], [11, 5], [13, 9]]
# [3, 5], [5,9], [3, 3], [7, 5], [3, 7], [5, 5]  for fmnist
# [9, 5], [7, 5], [11, 5], [5, 7], [9, 3], [9, 5], [3, 9], [7, 7], [11, 7]
for squash_value in range(testRange):
    my_scores.append([])
    my_rects.append([])
    y_train = copy.copy(y_train_i)
    y_test = copy.copy(y_test_i)

    for index in range(len(y_train)):
        if y_train[index] != zucchini[squash_value]:
            y_train[index] = 1
        else:
            y_train[index] = 0

    for index in range(len(y_test)):
        if y_test[index] != zucchini[squash_value]:
            y_test[index] = 1
        else:
            y_test[index] = 0

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    myConvolutions = []
    print('\ntesting ' + str(zucchini[squash_value]))
    for (i, j) in rects:
            print(i, j)
            core1 = doubleConvMP(insideCores, (i, j), input1)
            core1 = Dropout(0.25)(core1)
            core1 = Dense(128, activation='relu')(core1)
            core1 = Dropout(0.5)(core1)
            core1 = Dense(num_classes, activation='softmax')(core1)

            model = Model(input1, core1)
            model.compile(loss=h.loss_factory((-1.0, 0.5)),  # =keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(lr=myLr),
                          metrics=['accuracy'])
            for local_batch_size in first_training_batch_sizes:
                model.fit(x_train, y_train,
                          batch_size=local_batch_size,
                          epochs=first_epochs,
                          verbose=initial_verbosity,
                          validation_data=None)  # (x_test, y_test))
            wewe = x_test
            my_scores[squash_value].append(model.evaluate(x_test, y_test, verbose=0, batch_size=512)[1])
            my_rects[squash_value].append([i, j])
    the_index = np.argmax(my_scores[squash_value])
    print('  Test accuracy:' + str(my_scores[squash_value][the_index]) +
          ' for ' + str(my_rects[squash_value][the_index]) + '\n' + str(np.round(my_scores[squash_value], 4)))