import keras
import numpy as np
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import activations, Lambda
from keras.preprocessing.image import ImageDataGenerator as Im
from keras import backend as K
import copy
from keras.engine.topology import Layer
import tensorflow as tf
import os
from time import localtime, strftime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
cd Desk*/ker*
python hLayers.py
'''


class removeDim0(Layer):
    def __init__(self, activation=None, **kwargs):
        self.activation = activations.get(activation)
        super(removeDim0, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.trainable = False
        super(removeDim0, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        newx = x[:, ::2]
        if self.activation is not None:
            newx = self.activation(newx)
        return newx

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]/2


class rotateImage(Layer):

    def __init__(self, rotationAngle=0, **kwargs):
        self.rotationAngle = rotationAngle
        super(rotateImage, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.trainable = False
        super(rotateImage, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        if self.rotationAngle == 0:
            newx = x
        else:
            newx = tf.contrib.image.rotate(
                x,
                5,
                interpolation='NEAREST',
                name=None
                )
        return newx

    def compute_output_shape(self, input_shape):
        return input_shape


class roundInputs(Layer):

    def __init__(self, **kwargs):
        super(roundInputs, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.trainable = False
        super(roundInputs, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        newx = K.round(x)
        return newx

    def compute_output_shape(self, input_shape):
        return input_shape


class twoHotFilters(Layer):

    def __init__(self, **kwargs):
        self.filters = []
        super(twoHotFilters, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(self.filters) == 0:
            inputLength = input_shape[-1]
            for j in range(inputLength):
                for k in range(1, inputLength-j):
                    x = K.zeros(inputLength)
                    x[j] = 1.0
                    x[j+k] = 1.0
                    # x = K.variable(value=x, dtype='float32', name='kernel'+str(j)+str(k))
                    self.filters.append(x)
                    # if j==0 and k==1:
                    #     print (x)
            self.filters = tf.transpose(self.filters)
            self.set_weights(K.variable(value=self.filters, dtype='float32', name='filters').reshape(input_shape))
            self.trainable_weights.append(self.weights)
        # self.kernel = self.add_weight(name='kernel',
        #                                 shape=(input_shape[1], self.output_dim),
        #                                 initializer='zeros',
        #                                 trainable=True)
        # print(self.kernel)
        # print(self.filters)
        # self.kernel = self.filters
        super(twoHotFilters, self).build(input_shape)

    def call(self, inputs, **kwargs):
        l_inputs = K.dot(inputs, self.weights)
        return l_inputs

    def compute_output_shape(self, input_shape):
        out_size = input_shape[1]*(input_shape[1]-1)/2
        return input_shape[0], out_size


class oneHotFilters(Layer):

    def __init__(self, activation=None, **kwargs):
        self.activation = activations.get(activation)
        self.filters = []
        super(oneHotFilters, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(self.filters) == 0:
            inputLength = input_shape[-1]
            x = np.zeros((inputLength, inputLength), dtype='float32')
            for j in range(inputLength):
                x[j][j] = 1.0
            # b = np.zeros(inputLength)
            self.set_weights(x)
            # self.kernel=K.variable(value=x, dtype='float32', name='filters')
            self.trainable_weights.append(self.weights)
        super(oneHotFilters, self).build(input_shape)

    def call(self, inputs, **kwargs):
        l_inputs = K.dot(inputs, self.weights)
        if self.activation is not None:
            l_inputs = self.activation(l_inputs)
        return l_inputs

    def compute_output_shape(self, input_shape):
        return input_shape


def printTime():
    print('\n'+strftime("%H:%M:%S", localtime()))


def oneHotDense(myDim, theInput):
    x = np.zeros((myDim, myDim), dtype='float32')
    for j in range(myDim):
        x[j][j] = 1.0
    theCore = Dense(myDim, activation='softmax', use_bias=False, weights=[x])(theInput)
    return theCore


def removeDimDense(myDim, theInput):
    x = np.zeros((myDim, myDim/2), dtype='float32')
    for j in range(myDim/2):
        x[j*2][j] = 1.0
    theCore = Dense(myDim/2, activation='softmax', use_bias=False, weights=[x], trainable=False)(theInput)
    return theCore


def getHmnistFlat():
    (x_train, y_train_i), (x_test, y_test_i) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train_i), (x_test, y_test_i)


def getHmnistConv():
    (x_train, y_train_i), (x_test, y_test_i) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train_i), (x_test, y_test_i)


def getHmnistConvQ():
    (x_train, y_train_i), (x_test, y_test_i) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 14, 56, 1)
    x_test = x_test.reshape(x_test.shape[0], 14, 56, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train_i), (x_test, y_test_i)


def getHFmnistFlat():
    (x_train, y_train_i), (x_test, y_test_i) = fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train_i), (x_test, y_test_i)


def getHFmnistConv():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def getHFmnistConvNoise(noiseImagesNumber):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    data = np.random.randint(0, high=25, size=(noiseImagesNumber, 28, 28, 1), dtype='int').astype('float32')
    # labels = np.random.randint(0, high=10, size=noiseImagesNumber, dtype='int')
    labels = y_train[0:noiseImagesNumber]
    xint = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    data = np.maximum(xint[0:noiseImagesNumber, :, :, :] + data, 255.0)
    x_train = np.concatenate([xint, data], axis=0)
    y_train = np.append(y_train, labels)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def getHmnistConvNoise(noiseImagesNumber):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    data = np.random.randint(0, high=25, size=(noiseImagesNumber, 28, 28, 1), dtype='int').astype('float32')
    # labels = np.random.randint(0, high=10, size=noiseImagesNumber, dtype='int')
    labels = y_train[0:noiseImagesNumber]
    xint = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    data = np.maximum(xint[0:noiseImagesNumber, :, :, :] + data, 255.0)
    x_train = np.concatenate([xint, data], axis=0)
    y_train = np.append(y_train, labels)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def getHFmnistConvQ():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 14, 56, 1)
    x_test = x_test.reshape(x_test.shape[0], 14, 56, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def getHcifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
    x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def getHcifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
    x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def removeEveryOther_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] /= 2
    return tuple(shape)


def removeEveryOtherCore(x):
    halfx = x[:, ::2]
    return halfx


def removeEveryOther(x):
    return Lambda(removeEveryOtherCore, output_shape=removeEveryOther_output_shape)(x)


def doubleConvMP(numCores, (width, height), complementSize, myLayers, myInput):
    conv1 = Conv2D(numCores, (width, height), padding='valid', activation='relu')(myInput)
    myLayers.append(conv1)
    conv1 = Conv2D(4*numCores, (complementSize-width, complementSize-height), padding='valid', activation='relu')(conv1)
    myLayers.append(conv1)
    conv1 = MaxPooling2D((2, 2), strides=(1, 1), padding='valid')(conv1)
    return conv1


def seeded_initializer_BW(theShape):
    if len(theShape) != 4:
        raise ValueError('wrong shape size: expected 4 but got', len(theShape))
    output = K.variable(K.round(K.random_normal(theShape, mean=0.7, stddev=0.4)))
    # for channelOut in range(theShape[3]):
    #     for channelIn in range(theShape[2]):
    #         for x in range
    return output


# (x_train, y_train), (x_test, y_test) = getHmnistConv()
#
#
#
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
#
# input1 = Input(shape=[28,28,1])
# test = rotateImage(-5)(input1)
# test = Flatten()(test)
# test = Dense(128,activation='relu')(test)
# test = Dense(64,activation='relu')(test)
# test2 = Dense(10,activation='softmax')(test)
#
# # test1 = twoHotFilters()(test2)
# # test = Dropout(0.25)(test)
# # test = Dense(128,activation='relu')(test1)
# # test = Dropout(0.25)(test)
# # test = Dense(64,activation='relu')(test)
# # test = Dense(10,activation='softmax')(test)
# model = Model(inputs=input1, outputs=test2)
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           batch_size=128,
#           epochs=2,
#           verbose=1,
#           validation_data=(x_test, y_test) )
