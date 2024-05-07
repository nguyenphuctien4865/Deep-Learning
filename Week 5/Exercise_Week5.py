import numpy as np
import tensorflow as tf
import keras
#Load a dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


#Cifar-10 dataset : [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck] 

def split_dataset(X, y):
    y_4_or_5 = (y == 4) | (y == 9) # 4: deer, 5: dog
    y_A = y[~y_4_or_5.flatten()]
    y_A[y_A > 4] -= 2 # class indices 5, 6, 7, 8 are moved to 3, 4, 5, 6
    y_B = (y[y_4_or_5.flatten()] == 4).astype(np.float32) # binary classification task: is it a deer (class 4)?
    
    return ((X[~y_4_or_5.flatten()], y_A),
            (X[y_4_or_5.flatten()], y_B))
    
(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)

#Model A
model_A = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(4, activation="softmax")
])
