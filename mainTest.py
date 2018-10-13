import keras
import hLayers as h

# from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Input
# from keras.layers import activations
# from keras import backend
import copy
# from keras.engine.topology import Layer
# import tensorflow as tf


'''
tensorboard --logdir=/Users/hvelde/Desktop/keras/logs
http://localhost:6006
cd Desk*/ker*
python mainTest.py
'''


def removeDim_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] /= 2
    return tuple(shape)


def myLambda(x):
    halfx = x[:, ::2]
    return halfx


# def doubleConvMP(numCores, (width, height), complementSize, myLayers, myInput):
#     conv1= Conv2D(numCores,(width,height), padding='valid', activation='relu')(myInput)
#     myLayers.append(conv1)
#     conv1 = Conv2D(4*numCores,(complementSize-width,
#                    complementSize-height), padding='valid', activation='relu')(conv1)
#     myLayers.append(conv1)
#     conv1 = MaxPooling2D((2,2), strides=(1,1), padding='valid')(conv1)
#     return conv1


def doubleConvMP(numCores, (width, height), endLayers, myInput):
    conv1 = Conv2D(numCores, (width, height), data_format=channel_style, padding='valid', activation='relu')(myInput)
    endLayers.append(conv1)
    conv1 = Conv2D(2*numCores, (3, 3), data_format=channel_style, padding='valid', activation='relu')(conv1)
    endLayers.append(conv1)
    conv1 = Conv2D(4*numCores, (height, width), data_format=channel_style, padding='valid', activation='relu')(conv1)
    endLayers.append(conv1)
    conv1 = MaxPooling2D((2, 2), data_format=channel_style, strides=(1, 1), padding='valid')(conv1)
    conv1 = Dropout(0.15)(conv1)
    # conv1 = Flatten()(conv1)
    return conv1


# data_set_type = 'cifar'
# data_set_type = 'cifar100'
data_set_type = 'mnist'
# data_set_type = 'fashion_mnist'
# data_set_type = 'fashion_mnist_noise'
# data_set_type = 'mnist_noise'
noiseImagesNumber = 6000

print('\n' + data_set_type)
num_classes = 2
first_epochs = 1
first_training_batch_sizes = [128, 256, 512, 1024]
final_epochs = 2
final_training_batch_sizes = [512, 1024]
initial_verbosity = 1
final_verbosity = 1
testRange = 10

kernel_sizes = [(2, 6), (6, 2), (5, 3), (3, 5), (4, 4)]
insideCores = 1

print('\nlabels tested: '+str(testRange))

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

myLayers = []
input1 = Input(shape=my_shape)

for squash_value in range(testRange):
    localLayers = []
    myLayers.append(localLayers)
    y_train = copy.copy(y_train_i)
    y_test = copy.copy(y_test_i)

    for index in range(len(y_train)):
        if y_train[index] != squash_value:
            y_train[index] = 1
        else:
            y_train[index] = 0

    for index in range(len(y_test)):
        if y_test[index] != squash_value:
            y_test[index] = 1
        else:
            y_test[index] = 0

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    myConvolutions = []
    for aSize in kernel_sizes:
        myConvolutions.append(doubleConvMP(insideCores, aSize, myLayers[squash_value], input1))
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
    myLr = 0.001
    print('\ntesting '+str(squash_value))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=myLr),
                  metrics=['accuracy'])

    tb = keras.callbacks.TensorBoard(log_dir='./logs/main_test_mnist_logs_' + str(squash_value),
                                     histogram_freq=1,
                                     batch_size=128,
                                     write_graph=False,
                                     write_grads=False,
                                     write_images=True,
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None,
                                     embeddings_data=None,
                                     update_freq='batch')

    for local_batch_size in first_training_batch_sizes:
        tb.batch_size = local_batch_size
        model.fit(x_train, y_train,
                  batch_size=local_batch_size,
                  epochs=first_epochs,
                  verbose=initial_verbosity,
                  callbacks=[tb],
                  validation_data=(x_test, y_test))

endpoints = []

for i in range(testRange):
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

# core2Int = h.removeDimDense(2*endRange,core2)
core22 = Lambda(myLambda, output_shape=removeDim_output_shape)(core2)

# core2 = h.twoHotFilters()(core22)

# core2 = keras.layers.concatenate([core22, core2])

# core2 = h.oneHotDense(testRange, core22)
# core2 = Dropout(0.25)(core2)

core2 = Dense(endRange, activation=None, use_bias=False, kernel_initializer='identity')(core22)


if endRange == 10:
    model3 = Model(input1, core22)
    model3.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adam(),
                   metrics=['accuracy'])
    print('\n\nModel 1 just the score')
    score = model3.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('')
print('\n\nTesting all values, model 2 ~ oneHot')

model2 = Model(input1, core2)
model2.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adam(lr=0.00001),
               metrics=['accuracy'])

tb = keras.callbacks.TensorBoard(log_dir='./logs/main_test_mnist_logs_main',
                                 histogram_freq=1,
                                 batch_size=128,
                                 write_graph=True,
                                 write_grads=True,
                                 write_images=True,
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None,
                                 embeddings_data=None,
                                 update_freq='batch')

for local_batch_size in final_training_batch_sizes:
    tb.batch_size = local_batch_size
    model2.fit(x_train, y_train,
               batch_size=local_batch_size,
               epochs=final_epochs,
               verbose=final_verbosity,
               callbacks=[tb],
               validation_data=(x_test, y_test))
    if local_batch_size == 128:
        for aLayer in model2.layers:
            aLayer.trainable = True

h.printTime()
