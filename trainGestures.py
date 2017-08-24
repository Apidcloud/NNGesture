import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
import random
import math
import os

data = []

n_classes = 0

for file in os.listdir("imgDataObjects/"):
    if file.endswith(".npy"):
    	tempData = np.load(os.path.join("imgDataObjects", file))
    	for i in range(0,len(tempData)):
    		data.append({'image':tempData[i], 'label' : n_classes })
    	n_classes += 1

random.shuffle(data)

# training data
trainData = np.empty([len(data), 784])

# labels, bushes = 0, trees = 1, fence = 2, etc
trainLabels = np.empty([len(data)])

for i in range(len(data)):
	trainData[i] = data[i]['image']
	trainLabels[i] = data[i]['label']


# reshape data
trainData = trainData.reshape(len(data), 28, 28,1).astype('float32')

# normalize from [0, 255] to [0, 1]
trainData /= 255

# convert integer labels into one-hot vectors
trainLabels = to_categorical(trainLabels, n_classes)

# start building the network
model = Sequential()

n_filters = 32

n_conv = 3

n_pool = 2

# model v1
# model.add(Convolution2D(
#         n_filters, 
#         kernel_size=(n_conv, n_conv),
#         # we have a 28x28 single channel (grayscale) image
#         # so the input shape should be (28, 28, 1)
#         input_shape=(28, 28, 1)
# ))
# model.add(Activation('relu')) #what does activation function do again

# model.add(Convolution2D(n_filters, kernel_size=(n_conv, n_conv)))

# model.add(Activation('relu'))

# # then we apply pooling to summarize the features
# # extracted thus far
# model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))


# model.add(Dropout(0.25))

# # flatten the data for the 1D layers
# model.add(Flatten())

# # Dense(n_outputs)
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# # the softmax output layer gives us a probablity for each class
# model.add(Dense(n_classes))
# model.add(Activation('softmax'))

# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )

# model = Sequential()
# model.add(Conv2D(, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))


# model v2
model.add(Convolution2D(
        n_filters, 
        kernel_size=(n_conv, n_conv),
        # we have a 28x28 single channel (grayscale) image
        # so the input shape should be (28, 28, 1)
        input_shape=(28, 28, 1)
))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# how many examples to look at during each update step
batch_size = 128

# how many times to run through the full set of examples
n_epochs = 30

# the training may be slow depending on your computer
model.fit(trainData,
          trainLabels,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_split=0.3)

# loss, accuracy = model.evaluate(testData, testLabels, verbose=0)


# print('loss:', loss)
# print('accuracy:', accuracy)

# single prediction
# predictData = np.empty([1, 28, 28, 1])
# index = random.randint(0,len(testData))
# predictData[0] = testData[index]
# prediction = model.predict(predictData, batch_size=1, verbose=1)
# print(prediction[0].round(2))
# plt.imshow(np.reshape(predictData[0],[28,28]), interpolation="nearest", cmap="gray")

# save model so we don't have to train again!
model_json = model.to_json()
with open("savedModel/model_objects.json", "w") as json_file: json_file.write(model_json)
model.save_weights("savedModel/model_objects.h5")