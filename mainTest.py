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
python mainTest.py
'''


# data_set_type = 'cifar'
# data_set_type = 'cifar100'
# data_set_type = 'mnist'
# data_set_type = 'mnist_noise'
data_set_type = 'fashion_mnist'
# data_set_type = 'fashion_mnist_noise'
noiseImagesNumber = 10000

save_weights = False
reload_weights = not save_weights
weights_path = 'weights/main_test_' + data_set_type + '.h5'

save_model1_results = False
load_model1_results = not save_model1_results


def removeDim_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] /= 2
    return tuple(shape)


def myLambda(x):
    halfx = x[:, ::2]
    return halfx


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


print('\n' + data_set_type)
num_classes = 2
first_epochs = 1
first_training_batch_sizes = [64, 128, 256, 512, 1024]
final_epochs = 5
final_training_batch_sizes = [128, 512, 1024, 2048]
initial_verbosity = 1
final_verbosity = 1
testRange = 10

kernel_sizes = [(2, 6), (6, 2), (5, 3), (3, 5), (4, 4)]
insideCores = 3

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
    if save_weights:
        print('\ntesting ' + str(squash_value))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=myLr),
                      metrics=['accuracy'])

        # tb = keras.callbacks.TensorBoard(log_dir='./logs/main_test_' + str(data_set_type) +
        #                                          '_logs_' + str(squash_value),
        #                                  histogram_freq=1,
        #                                  batch_size=128,
        #                                  write_graph=False,
        #                                  write_grads=False,
        #                                  write_images=True,
        #                                  embeddings_freq=0,
        #                                  embeddings_layer_names=None,
        #                                  embeddings_metadata=None,
        #                                  embeddings_data=None,
        #                                  update_freq='batch')
        '''
        #   While it is customary to change the learning rate between epochs, there are constraints to the
        #   maneuver that may be seen as disadvantages. 
        #   When one uses an optimizer such as Adam, the learning rate in effect varies at every epoch since
        #   there is a dynamic dampening of the gradient vector based in part on age, and of course, there is a 
        #   momentum component applied to the learning rate. So the choice of a new learning rate has deeper 
        #   implications than just slowing down learning by a known factor. That kind of guess work can be felt 
        #   to be sub-optimal.
        #   Additionally, in Keras, a manual change in learning rate requires re-compilation of the model. That appears 
        #   to reset a number of parameters, again without much visibility into the matter.
        #   
        #   The approach taken here consists in modifying the batch size rather than the learning rate. The concept
        #   becomes much more intuitive and we are not interrupting the chosen fluctuations in parameters. By doubling 
        #   the size of the batch, one reduces the granularity of the learned features since these are averaged over 
        #   the batches. Hence one directly and measurably reduces over-fitting. More importantly, this greatly
        #   accelerates convergence of the model towards its capacity.
        #
        '''
        for local_batch_size in first_training_batch_sizes:
            # tb.batch_size = local_batch_size
            model.fit(x_train, y_train,
                      batch_size=local_batch_size,
                      epochs=first_epochs,
                      verbose=initial_verbosity,
                      # callbacks=[tb],
                      validation_data=None)  # (x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss: ' + str(score[0]) + '  Test accuracy:' + str(score[1]))

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


input2 = Input(shape=[10])

inputLength = testRange
output_length = inputLength * (inputLength - 1) / 2
my_index = -1

#  combination filter coefficient value
filter_coefficient = 0.5

two_hot_array = np.random.randn(inputLength, output_length) / 10

for j in range(inputLength):
    for k in range(1, inputLength - j):
        my_index += 1
        two_hot_array[j, my_index] = filter_coefficient
        two_hot_array[j + k, my_index] = filter_coefficient
filters = [two_hot_array]

core2 = Dense(output_length, activation=None, use_bias=False, weights=filters, name='hdense')(input2)

core2 = keras.layers.concatenate([input2, core2])

core3 = Reshape([55, 1])(core2)
core3 = Conv1D(64, 12)(core3)
core3 = Flatten()(core3)
core2 = keras.layers.concatenate([core3, core2])

'''  MNIST baseline is 9600 at first epoch end with a random filter and no bias on the last full dense.  '''


last_dense_layer_output_size = 256

core2 = Dense(last_dense_layer_output_size, activation='relu', use_bias=False)(core2)
core2 = Dropout(0.15)(core2)
core2 = Dense(2*last_dense_layer_output_size, activation='relu')(core2)
core2 = Dropout(0.15)(core2)
core2 = Dense(endRange, activation='softmax', use_bias=False, name='finalDense')(core2)


model1 = Model(input1, core22)
if reload_weights:
    print('loading weights from ' + weights_path)
    model1.load_weights(weights_path)
model1.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adam(),
               metrics=['accuracy'])
if save_weights:
    print('\n\nModel 1 just the score')
    score = model1.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('saving weights at ' + weights_path)
    model1.save_weights(weights_path)
print('')

if not load_model1_results:
    print('evaluating inputs (train & test) on first model')
    x_train = model1.predict(x_train, verbose=1, batch_size=512)
    x_test = model1.predict(x_test, verbose=1, batch_size=512)
    if save_model1_results:
        np.save('weights/model1_output_train.npy', x_train)
        np.save('weights/model1_output_test.npy', x_test)
else:
    x_train = np.load('weights/model1_output_' + data_set_type + '_train.npy')
    x_test = np.load('weights/model1_output_' + data_set_type + '_test.npy')


print('Testing all values, model 2 ~ twoHot')
last_lr = 0.0001
print(str(last_lr)+' lr :: filter at '+str(filter_coefficient))
model2 = Model(input2, core2)
model2.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adam(lr=last_lr),
               metrics=['accuracy'])

tb = keras.callbacks.TensorBoard(log_dir='./logs/main_test_' + str(data_set_type) + '_logs_main',
                                 histogram_freq=1,
                                 batch_size=128,
                                 write_graph=False,
                                 write_grads=True,
                                 write_images=True,
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None,
                                 embeddings_data=None,
                                 update_freq='batch')

for local_batch_size in final_training_batch_sizes:
    tb.histogram_freq = 16*128/local_batch_size  # try not to capture data too often
    tb.batch_size = local_batch_size  # ze trick of adjusting ze batch sizes instead of the lr
    print('batch: '+str(local_batch_size))
    model2.fit(x_train, y_train,
               batch_size=local_batch_size,
               epochs=final_epochs,
               verbose=final_verbosity,
               # callbacks=[tb],
               validation_data=(x_test, y_test))
    if local_batch_size == 1024:
        for aLayer in model2.layers:
            aLayer.trainable = True

h.printTime()
