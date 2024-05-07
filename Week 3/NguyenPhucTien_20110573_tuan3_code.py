### Nguyễn Phúc Tiền - 20110573

import tensorflow as tf 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
import numpy as np
####### Exc 1 - W3 #########
                    #### Exc3-W1 ####

model= Sequential([
    Flatten(input_shape=(50,50,3)),
    Dense(128,activation='relu'),
    Dense(2,activation='softmax')
])

model.compile(optimizer='adam',
              loss='space_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


                    #### Exc4-W1 ####

model = Sequential([
    Flatten(input_shape=(50, 50)), 
    Dense(128, activation='relu'),  
    Dense(5, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


                    #### Exc5-W1 ####

model = Sequential([
    Dense(512, activation='relu', input_shape=(640,)),  
    Dense(256, activation='relu'),  
    Dense(5, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#######   Exc2 - W3  #########

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images_filtered = []
test_images_filtered = []
train_labels_filtered = []
test_labels_filtered = []

for i in range(len(train_images)):
    if train_labels[i] == 7 or train_labels[i] == 8:
        train_images_filtered.append(train_images[i])
        train_labels_filtered.append(train_labels[i])

for i in range(len(test_images)):
    if test_labels[i] == 7 or test_labels[i] == 8:
        test_images_filtered.append(test_images[i])
        test_labels_filtered.append(test_labels[i])

train_images_filtered = tf.expand_dims(train_images_filtered, axis=-1) / 255
test_images_filtered = tf.expand_dims(test_images_filtered, axis=-1) / 255

train_images_filtered = train_images_filtered.numpy()
test_images_filtered = test_images_filtered.numpy()
train_labels_filtered = np.array(train_labels_filtered)
test_labels_filtered = np.array(test_labels_filtered)

model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_images_filtered, train_labels_filtered, epochs=5, batch_size=32)

test_loss, test_acc = model.evaluate(test_images_filtered, test_labels_filtered)
print('Test accuracy:', test_acc)

## Video: https://drive.google.com/file/d/1nk8cpAgSpYxHEaXieZeCIm6Jmzj1EMQp/view?usp=sharing
