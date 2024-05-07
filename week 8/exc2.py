import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset, info = tfds.load('iris', split='train', with_info=True)

# Convert the dataset to numpy arrays
data = np.array([example['features'].numpy() for example in dataset])
labels = np.array([example['label'].numpy() for example in dataset])

# Split features and categorical labels
X = data[:, :-1]  # Features
y_cat = labels.astype('int32')  # Categorical labels

# Define the neural network architecture for embedding
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer for categorical labels
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y_cat, epochs=50, validation_split=0.2, verbose=0)

# Extract embeddings
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
embeddings = embedding_model.predict(X)

# Plot the embeddings
plt.figure(figsize=(8, 8))
for i in range(3):  # There are 3 categories in iris dataset
    plt.scatter(embeddings[y_cat == i, 0], embeddings[y_cat == i, 1], label=f'Category {i}')
plt.title('Embeddings Visualization')
plt.xlabel('Embedding Dimension 1')
plt.ylabel('Embedding Dimension 2')
plt.legend()
plt.show()
