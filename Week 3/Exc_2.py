import tensorflow as tf
import tensorflow_datasets as tfds

# Load the MNIST dataset
datasets, info = tfds.load('mnist', with_info=True, as_supervised=True,
                          split=['train', 'test'])

# Filter out examples that are not 7 or 8
train_dataset = datasets[0].filter(lambda x, y: tf.math.equal(y, 7) or tf.math.equal(y, 8))
test_dataset = datasets[1].filter(lambda x, y: tf.math.equal(y, 7) or tf.math.equal(y, 8))

# Create a training and test dataset
# train_dataset, test_dataset = datasets

# Preprocess the images
train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

# Create a buffer size to randomly shuffle the data
BUFFER_SIZE = 10000

# Create a train dataset with a buffer size
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(32)

# Create a test dataset with a batch size
test_dataset = test_dataset.batch(32)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)