import keras
import hLayers as h
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, Input, Conv1D
# from keras.layers import activations
# from keras import backend
import copy
# from keras.engine.topology import Layer
# import tensorflow as tf


'''
tensorboard --logdir=/Users/hvelde/Desktop/keras/logs
http://localhost:6006
cd Desk*/ker*
python catClass.py


baseline on 9: 9974
kernel_sizes = [(3, 14), (14, 3), (10, 5), (5, 10), (8, 8)]
insideCores = 2
1.0 + 0.5 / 1.0 - 0.5
'''


# data_set_type = 'cifar'
# data_set_type = 'cifar100'
data_set_type = 'mnist'
# data_set_type = 'mnist_noise'
# data_set_type = 'fashion_mnist'
# data_set_type = 'fashion_mnist_noise'
noiseImagesNumber = 10000

save_weights = True
reload_weights = not save_weights
weights_path = 'weights/large_conv_test_' + data_set_type + '.h5'

save_model1_results = True
load_model1_results = not save_model1_results


def removeDim_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] /= 2
    return tuple(shape)


def myLambda(x):
    halfx = x[:, ::2]
    return halfx


def doubleConvMP(numCores, (width, height), endLayers, myInput, my_min_core_size):
    conv1 = Conv2D(numCores, (width, height), data_format=channel_style, padding='valid', activation='relu')(myInput)
    endLayers.append(conv1)
    conv1 = Conv2D(2*numCores, (3, 3), data_format=channel_style, padding='valid', activation='relu')(conv1)
    endLayers.append(conv1)
    conv1 = Conv2D(4*numCores, (height, width), data_format=channel_style, padding='valid', activation='relu')(conv1)
    endLayers.append(conv1)
    adjustment = width + height - my_min_core_size
    assert adjustment > -1
    if adjustment > 0:
        adj1 = adjustment / 2
        adj2 = adj1 + adj1 % 2
        conv1 = keras.layers.ZeroPadding2D(padding=((adj1, adj2), (adj1, adj2)), data_format=channel_style)(conv1)
    conv1 = MaxPooling2D((2, 2), data_format=channel_style, padding='valid')(conv1)
    # conv1 = Flatten()(conv1)
    return conv1


print('\n' + data_set_type)
num_classes = 2
first_epochs = 2
first_training_batch_sizes = [64, 128, 256, 512, 1024]
final_epochs = 1
final_training_batch_sizes = [128, 256, 512, 1024]
initial_verbosity = 1
final_verbosity = 1
testRange = 10

kernel_sizes = [(13, 7), (7, 5), (5, 7), (3, 9), (9, 3), (2, 10), (6, 6)]
min_core_size = np.min(np.sum(kernel_sizes, -1))

insideCores = 1
resampleSize = [55, 55]

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

print('\nlabels tested: '+str(testRange))

h.printTime()

myLayers = []
input1 = Input(shape=my_shape)
# resizeLayer = h.resampleImage(resampleSize)(input1)
resizeLayer = input1

# zucchini = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
zucchini = [[0, 2, 4, 6, 8], [2, 3, 6, 7], [4, 5, 6, 7], [8, 9]]
myLr = 0.00075

for squash_value in range(len(zucchini)):
    localLayers = []
    myLayers.append(localLayers)
    y_train = copy.copy(y_train_i)
    y_test = copy.copy(y_test_i)

    for index in range(len(y_train)):
        if y_train[index] not in zucchini[squash_value]:
            y_train[index] = 1
        else:
            y_train[index] = 0

    for index in range(len(y_test)):
        if y_test[index] not in zucchini[squash_value]:
            y_test[index] = 1
        else:
            y_test[index] = 0

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    myConvolutions = []
    for aSize in kernel_sizes:
        myConvolutions.append(doubleConvMP(insideCores, aSize, myLayers[squash_value], resizeLayer, min_core_size))
    core1 = keras.layers.concatenate(myConvolutions)
    core1 = Dropout(0.25)(core1)
    core1 = Conv2D(64, (4, 4), padding='valid', activation='relu')(core1)
    myLayers[squash_value].append(core1)
    core1 = MaxPooling2D((2, 2), padding='valid')(core1)
    core1 = Flatten()(core1)
    core1 = Dropout(0.25)(core1)
    core1 = Dense(128, activation='relu')(core1)
    myLayers[squash_value].append(core1)
    core1 = Dropout(0.5)(core1)
    core1 = Dense(num_classes, activation='softmax')(core1)
    myLayers[squash_value].append(core1)

    model = Model(input1, core1)

    print('\ntesting ' + str(zucchini[squash_value]))
    for local_batch_size in first_training_batch_sizes:
        print(local_batch_size)
        # tb.batch_size = local_batch_size
        model.compile(loss=h.high_accuracy,  # =keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=myLr),
                      metrics=['accuracy'])
        model.fit(x_train, y_train,
                  batch_size=local_batch_size,
                  epochs=first_epochs,
                  verbose=initial_verbosity,
                  # callbacks=[tb],
                  validation_data=None)  # (x_test, y_test))

        # model.compile(loss=h.high_error,  # =keras.losses.categorical_crossentropy,
        #               optimizer=keras.optimizers.Adam(lr=myLr),
        #               metrics=['accuracy'])
        # model.fit(x_train, y_train,
        #           batch_size=local_batch_size,
        #           epochs=first_epochs,
        #           verbose=initial_verbosity,
        #           # callbacks=[tb],
        #           validation_data=None)  # (x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=1, batch_size=512)
    print('Test loss: ' + str(score[0]) + '  Test accuracy:' + str(score[1]))

endpoints = []

for i in range(len(zucchini)):
    llen = len(myLayers[i])
    for j in range(llen):
        myLayers[i][j].trainable = False
    endpoints.append(myLayers[i][llen-1])

y_train = copy.copy(y_train_i)
y_test = copy.copy(y_test_i)


if testRange == 10:
    endRange = testRange
else:
    endRange = testRange+1
    for index in range(len(y_train)):
        if y_train[index] >= testRange:
            y_train[index] = testRange
    for index in range(len(y_test)):
        if y_test[index] >= testRange:
            y_test[index] = testRange

y_train = keras.utils.to_categorical(y_train, endRange)
y_test = keras.utils.to_categorical(y_test, endRange)

core2 = keras.layers.concatenate(endpoints)

core22 = h.squareInput()(core2)
core2 = keras.layers.concatenate([core22, core2])

# core3 = Reshape([10, 2], name='reshape')(core2)
# core3 = Conv1D(64, 3, name='conv1d', data_format=channel_style)(core3)
# core3 = Flatten()(core3)
# core2 = keras.layers.concatenate([core3, core2])

last_dense_layer_output_size = 64

core2 = Dense(last_dense_layer_output_size, activation='relu', name='firstDense')(core2)
core2 = Dropout(0.10)(core2)
# core2 = Dense(2*last_dense_layer_output_size, activation='relu')(core2)
# core2 = Dropout(0.5)(core2)
core2 = Dense(endRange, activation='softmax', name='finalDense')(core2)


print('Testing all values, model 2')
last_lr = 0.001
mybeta = 0.99
print(str(last_lr)+' lr :: ')
model2 = Model(input1, core2)
model2.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adam(lr=last_lr, beta_2=mybeta),
               metrics=['accuracy'])

for local_batch_size in final_training_batch_sizes:
    print('batch: '+str(local_batch_size))
    model2.fit(x_train, y_train,
               batch_size=local_batch_size,
               epochs=final_epochs,
               verbose=final_verbosity,
               validation_data=(x_test, y_test))
    if local_batch_size >= 256:
        for aLayer in model2.layers:
            aLayer.trainable = True

h.printTime()
