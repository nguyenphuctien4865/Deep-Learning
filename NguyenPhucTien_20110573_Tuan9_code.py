# Last Edit: 2023-06-05
# Author: Nguyen Phuc Tien - 20110573

import tensorflow as tf
import tensorflow_datasets as tfds

dataset_name = "cifar10"
(train_data, test_data), info = tfds.load(name=dataset_name, 
                                          split=["train", "test"], 
                                          with_info=True)

def preprocess_data(data):
    image = tf.cast(data['image'], tf.float32) / 255.0  
    return image, data['label']

train_data = train_data.map(preprocess_data).shuffle(1000).batch(64)
test_data = test_data.map(preprocess_data).batch(64)

# Define LeNet model 
def create_lenet_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Define ResNet model 
def create_resnet_model():
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    for layer in base_model.layers:
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Compile and train models
def compile_and_train_model(model, train_data, test_data):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_data, epochs=5)
    test_loss, test_acc = model.evaluate(test_data)
    print("Test accuracy:", test_acc)

# LeNet
print("LeNet Model:")
lenet_model = create_lenet_model()
compile_and_train_model(lenet_model, train_data, test_data)

# ResNet
print("\nResNet Model:")
resnet_model = create_resnet_model()
compile_and_train_model(resnet_model, train_data, test_data)


## Drive video: https://drive.google.com/drive/folders/1hTl28J59zr1NK1vsnaXPOAFAcUlkvnaY?usp=sharing