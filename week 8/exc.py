# Import necessary libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import LearningRateScheduler

# Load Boston housing dataset
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Scale data features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Define neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Define a constant learning rate
def lr_schedule(epoch, lr):
    return 0.01

initial_learning_rate = 0.01
decay_steps = 1000
decay_rate = 0.96

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True
)

callback = LearningRateScheduler(lr_schedule)

# Compile model with mean absolute error (MAE) loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mae')

# Train model with batch size 32 and 100 epochs
history = model.fit(X_train_scaled, y_train,
                    validation_data=(X_valid_scaled, y_valid),
                    batch_size=64, epochs=100, callbacks=[callback],
                    verbose=2)

# Evaluate model on test data
test_loss = model.evaluate(X_test_scaled, y_test)
print(f'Test loss: {test_loss:.2f}')

# Fine-tune hyperparameters to achieve test loss < 3.0
# Try different architectures, batch sizes, learning rates, and number of epochs