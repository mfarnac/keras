import keras
import hLayers as h
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, SeparableConv1D
from keras.layers import Conv2D, MaxPooling2D, Input
# from keras.layers import activations, SeparableConv1D, SeparableConv2D
import numpy as np
'''
cd Desk*/ker*
python rectKernels_PNR_v1.py
'''


''' Choose the dataset '''
# data_set_type = 'cifar'
# data_set_type = 'cifar100'
# data_set_type = 'mnist'
data_set_type = 'fashion_mnist'
# data_set_type = 'fashion_mnist_noise'
# data_set_type = 'mnist_noise'
noiseImagesNumber = 6000

print('\n' + data_set_type)

num_classes = 10
channel_style = 'channels_last'
my_shape = [28, 28, 1]
if data_set_type == 'cifar':
    (x_train, y_train), (x_test, y_test) = h.getHcifar10()
    channel_style = 'channels_first'
    my_shape = [3, 32, 32]
elif data_set_type == 'cifar100':
    (x_train, y_train), (x_test, y_test) = h.getHcifar100()
    channel_style = 'channels_first'
    my_shape = [3, 32, 32]
    num_classes = 100
elif data_set_type == 'fashion_mnist':
    (x_train, y_train), (x_test, y_test) = h.getHFmnistConv()
elif data_set_type == 'fashion_mnist_noise':
    (x_train, y_train), (x_test, y_test) = h.getHFmnistConvNoise(noiseImagesNumber)
elif data_set_type == 'mnist_noise':
    (x_train, y_train), (x_test, y_test) = h.getHmnistConvNoise(noiseImagesNumber)
else:
    (x_train, y_train), (x_test, y_test) = h.getHmnistConv()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
''' End dataset parameter setting'''


def doubleConvMP(num_cores, asize, my_layers, my_input, my_min_core_size):
    localSize = (asize[0], asize[1])
    conv1 = Conv2D(num_cores, localSize, data_format=channel_style, padding='valid', activation='relu')(my_input)
    my_layers.append(conv1)
    # localSize = (3, 3)
    # conv1 = Conv2D(2 * num_cores, localSize, data_format=channel_style, padding='valid', activation='relu')(conv1)
    # my_layers.append(conv1)
    localSize = (asize[1], asize[0])
    conv1 = Conv2D(2 * num_cores, localSize, data_format=channel_style, padding='valid', activation='relu')(conv1)
    my_layers.append(conv1)
    adjustment = asize[0] + asize[1] - my_min_core_size
    assert adjustment > -1
    if adjustment > 0:
        adj1 = adjustment / 2
        adj2 = adj1 + adj1 % 2
        conv1 = keras.layers.ZeroPadding2D(padding=((adj1, adj2), (adj1, adj2)), data_format=channel_style)(conv1)
    localSize = (2, 2)
    conv1 = MaxPooling2D(localSize, data_format=channel_style, padding='valid')(conv1)
    return conv1


epochs = 3
h_batch_sizes = [64, 128, 256, 512, 1024, 2048]
numberOfModels = 4
insideCores = 3  # each convolution setup has this number of cores in the first layer and twice that in the second

core_sizes = [(13, 7), (7, 5), (5, 7), (3, 9), (9, 3), (2, 10)]
if data_set_type == 'fashion_mnist':
    core_sizes = [[3, 3], [3, 7], [7, 5], [5, 9]]
if data_set_type == 'cifar':
    core_sizes = [[9, 5], [11, 5], [5, 7], [9, 3], [3, 9], [7, 7]]
min_core_size = np.min(np.sum(core_sizes, -1))

print(str(numberOfModels)+' models; minimodel batch sizes: ' + str(h_batch_sizes) +
      ' - ' + str(insideCores) + ' cores: '+str(core_sizes))

miniModels = []
endpoints = []
input1 = Input(shape=my_shape)
h.printTime()

losses = [h.high_accuracy, h.high_error, h.my_loss_function, keras.losses.categorical_crossentropy]
''' Enter 1.0 for accuracy emphasis or -1.0 for loss emphasis, followed by the spread'''
loss_details = [(1.0, 0.2), (-1.0, 0.3), (1.0, 0.8), (-1.0, 0.1)]

for modelIndex in range(numberOfModels):  # number of models being run
    myConvolutions = []
    my_loss = h.loss_factory(loss_details[modelIndex])
    for aSize in core_sizes:
        myConvolutions.append(doubleConvMP(insideCores, aSize, endpoints, input1, min_core_size))
    core1 = keras.layers.concatenate(myConvolutions)
    core1 = Dropout(0.2)(core1)
    core1 = Conv2D(64, (3, 3), data_format=channel_style, padding='valid', activation='relu')(core1)
    endpoints.append(core1)
    core1 = Conv2D(128, (3, 3), data_format=channel_style, padding='valid', activation='relu')(core1)
    endpoints.append(core1)
    core1 = MaxPooling2D((2, 2), data_format=channel_style, padding='valid')(core1)
    core1 = Dropout(0.2)(core1)
    # miniModels.append(core1)
    core1 = Flatten()(core1)

    core1 = Dense(128, activation='relu')(core1)
    endpoints.append(core1)
    miniModels.append(core1)
    core1 = Dropout(0.25)(core1)
    core1 = Dense(num_classes, activation='softmax')(core1)
    endpoints.append(core1)
    model = Model(input1, core1)
    beta_1 = 0.9
    beta_2 = 0.99
    beta1 = beta_1
    beta2 = beta_2
    mylr = 0.001

    print('\nMinimodel ' + str(modelIndex + 1))
    for local_batch_size in h_batch_sizes:
        print(local_batch_size)
        model.compile(loss=my_loss,
                      optimizer=keras.optimizers.Adam(lr=mylr),
                      metrics=['accuracy'])
        model.fit(x_train, y_train,
                  batch_size=local_batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=None)
        mylr *= 0.9
    score = np.round(model.evaluate(x_test, y_test, batch_size=512, verbose=1), 4)
    print(score[1])

for aLayer in endpoints:
    aLayer.trainable = False

core2 = keras.layers.concatenate(miniModels)
core2 = keras.layers.Reshape([128, numberOfModels])(core2)
core2 = SeparableConv1D(64, 1, data_format=channel_style, padding='valid', activation='relu')(core2)
core2 = Flatten()(core2)
core2 = Dense(128, activation='relu')(core2)
core2 = Dropout(0.25)(core2)
core2 = Dense(num_classes, activation='softmax')(core2)
model = Model(input1, core2)

h_batch_sizes = [128, 256, 512, 1024, 1024, 2048, 2048]

print('\nFull model - batch sizes: ' + str(h_batch_sizes))
chit = 0
beta_1 = 0.81
beta_2 = 0.98
beta1 = beta_1
beta2 = beta_2
mylr = 0.0005
final_epochs = 2
for local_batch_size in h_batch_sizes:
    print(local_batch_size)
    model.compile(loss=h.loss_factory((1.0, 0.5)),
                  optimizer=keras.optimizers.Adam(lr=mylr),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=local_batch_size,  # note the local variable here.
              epochs=final_epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    mylr *= 0.8
    # model.compile(loss=h.high_error,
    #               optimizer=keras.optimizers.Adam(lr=mylr),
    #               metrics=['accuracy'])
    # model.fit(x_train, y_train,
    #           batch_size=local_batch_size,  # note the local variable here.
    #           epochs=1,
    #           verbose=1,
    #           validation_data=(x_test, y_test))
    model.compile(loss=h.loss_factory((1.0, 0.25)),
                  optimizer=keras.optimizers.Adam(lr=mylr),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=local_batch_size,  # note the local variable here.
              epochs=final_epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    for aLayer in endpoints:
        aLayer.trainable = True
h.printTime()
print('')
