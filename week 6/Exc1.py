# Week 6/Exc1.py
# Author: Nguyen Phuc Tien - MSSV: 20110573

import tensorflow as tf
from keras import layers, models, optimizers
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dropout
import time

# Load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model
def create_model():
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(layers.Dense(10, activation='softmax'))
    return model

model_constant_lr = create_model()
model_exp_decay = create_model()
model_piecewise_constant = create_model()

model_constant_lr.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)
model_exp_decay.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

boundaries = [10000, 20000]
values = [0.001, 0.0005, 0.0001]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
model_piecewise_constant.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                                  loss='sparse_categorical_crossentropy',
                                  metrics=['accuracy'])

# Train the models
epochs = 10
batch_size = 32

# Constant learning rate
start_time =  time.time()
history_constant_lr = model_constant_lr.fit(train_images, train_labels,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            validation_data=(test_images, test_labels),
                                            verbose=2)

history_constant_lr_time = time.time() - start_time

# Exponential learning rate
start_time =  time.time()
history_exp_decay = model_exp_decay.fit(train_images, train_labels,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(test_images, test_labels),
                                        verbose=2)
history_exp_decay_time = time.time() - start_time

# Piecewise constant learning rate
start_time =  time.time()
history_piecewise_constant = model_piecewise_constant.fit(train_images, train_labels,
                                                          epochs=epochs,
                                                          batch_size=batch_size,
                                                          validation_data=(test_images, test_labels),
                                                          verbose=2)
history_piecewise_constant_time = time.time() - start_time

# Save the models5
model_constant_lr.save('model_constant_lr.h5')
model_exp_decay.save('model_exp_decay.h5')
model_piecewise_constant.save('model_piecewise_constant.h5')

print("Constant learning rate: ", history_constant_lr_time)
print("Accuracy of constant learning rate: ", history_constant_lr.history['accuracy'])
print("Accuracy of constant learning rate: ", history_constant_lr.history['val_accuracy'])
print("Exponential learning rate: ", history_exp_decay_time)
print("Accuracy of exponential learning rate: ", history_exp_decay.history['accuracy'])
print("Accuracy of exponential learning rate: ", history_exp_decay.history['val_accuracy'])
print("Piecewise constant learning rate: ", history_piecewise_constant_time)
print("Accuracy of piecewise constant learning rate: ", history_piecewise_constant.history['accuracy'])
print("Accuracy of piecewise constant learning rate: ", history_piecewise_constant.history['val_accuracy'])

# drive video: https://drive.google.com/drive/folders/1hCgCONVhjNI2bLjzfAWAKeF470nKm-39?usp=sharing