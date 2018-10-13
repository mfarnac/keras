import keras
import hLayers as h
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Input, Conv1D
from keras.layers import activations, SeparableConv1D, SeparableConv2D

'''
cd Desk*/ker*
python rectKernels_v2.py
'''

# data_set_type = 'cifar'
# data_set_type = 'cifar100'
# data_set_type = 'mnist'
# data_set_type = 'fashion_mnist'
# data_set_type = 'fashion_mnist_noise'
data_set_type = 'mnist_noise'
noiseImagesNumber = 6000

print('\n' + data_set_type)

insideCores = 3  # each convolution setup has this number of cores in the first layer and twice that in the second

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


def doubleConvMP(num_cores, asize, my_layers, my_input):
    localSize = (asize[0], asize[1])
    conv1 = Conv2D(num_cores, localSize, data_format=channel_style, padding='valid', activation='relu')(my_input)
    my_layers.append(conv1)
    localSize = (3, 3)
    conv1 = Conv2D(2 * num_cores, localSize, data_format=channel_style, padding='same', activation='relu')(conv1)
    my_layers.append(conv1)
    localSize = (asize[1], asize[0])
    conv1 = Conv2D(4 * num_cores, localSize, data_format=channel_style, padding='valid', activation='relu')(conv1)
    my_layers.append(conv1)
    localSize = (2, 2)
    localStride = (1, 1)
    conv1 = MaxPooling2D(localSize, data_format=channel_style, strides=localStride, padding='valid')(conv1)
    return conv1


epochs = 1
h_batch_sizes = [64, 96, 128, 160, 192]
print('minimodel batch sizes: ' + str(h_batch_sizes) + ' - ' + str(insideCores) + ' cores')

miniModels = []
endpoints = []
input1 = Input(shape=my_shape)
h.printTime()
for modelIndex in range(2):  # number of models being run
    myConvolutions = []
    for aSize in [(2, 6), (6, 2), (5, 3), (3, 5), (4, 4)]:
        myConvolutions.append(doubleConvMP(insideCores, aSize, endpoints, input1))
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

    print('Minimodel ' + str(modelIndex + 1))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=beta1, beta_2=beta2),
                  metrics=['accuracy'])
    for local_batch_size in h_batch_sizes:
        print(local_batch_size)
        model.fit(x_train, y_train,
                  batch_size=local_batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

for aLayer in endpoints:
    aLayer.trainable = False
core2 = keras.layers.concatenate(miniModels)
# core2 = Reshape((-1,1))(core2)
# core2 = Conv2D(128, (3, 3), data_format=channel_style, strides=1, padding='valid', activation='relu')(core2)
# core2 = Flatten()(core2)
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
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=beta1, beta_2=beta2),
              metrics=['accuracy'])

for local_batch_size in h_batch_sizes:
    print(local_batch_size)
    model.fit(x_train, y_train,
              batch_size=local_batch_size,  # note the local variable here.
              epochs=3,
              verbose=1,
              validation_data=(x_test, y_test))
    beta1 *= beta_1
    beta2 *= beta_1
    if local_batch_size >= 256:
        for aLayer in endpoints:
            aLayer.trainable = True
h.printTime()
print('')
