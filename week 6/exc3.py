import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam

# Load and preprocess the CIFAR-10 dataset
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
X_train = X_train / 255.0
X_valid = X_valid / 255.0
X_test = X_test / 255.0

# Build the model
model = Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20, 0, -1):
    model.add(Dense(2**_, kernel_initializer='he_normal', activation='elu'))
    model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

# Compile the model
optimizer = Nadam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define the callbacks
early_stopping_cb = EarlyStopping(patience=10)
onecycle = keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 10**(epoch / 20))

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[early_stopping_cb, onecycle])
history.save("exc3.h5")
# # Fine-tune the initial learning rate
# import numpy as np
# import matplotlib.pyplot as plt
# lrs = 0.001 * (10 ** (np.arange(100) / 20))
# plt.semilogx(lrs, history.history["loss"])
# plt.axis([0.001, 1, 0, np.max(history.history["loss"])])
# plt.xlabel("Learning Rate")
# plt.ylabel("Loss")
# plt.show()