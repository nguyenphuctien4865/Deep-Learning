import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Filter the data
train_mask = (y_train == 4) | (y_train == 5) 
test_mask = (y_test == 4) | (y_test == 5) 
x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

# # Define the model
# model = Sequential([
#     Flatten(input_shape=(28, 28)),
#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dense(10)
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # Define the callbacks
# checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
# early_stopping = EarlyStopping(patience=2)
# tensorboard = TensorBoard(log_dir='./logs')

# # Train the model
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),
#           callbacks=[checkpoint, early_stopping, tensorboard])

# # Evaluate the model
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
# print('\nTest accuracy:', test_acc)

##Video: https://drive.google.com/drive/folders/1z5-1elt5wYG2V2bCddhNOdfp6GMxE2LA?usp=sharing


# Load the model
model = load_model('best_model.h5')

# Now you can use this model to make predictions

import matplotlib.pyplot as plt
import numpy as np

# Make predictions
predictions = model.predict(x_test)

# Function to plot images and labels
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

# Plot the first X test images, their predicted labels, and the true labels.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, y_test, x_test)
plt.tight_layout()
plt.show()