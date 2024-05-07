#last edit: 10/06/2023
#Author: Nguyen Phuc Tien

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, datasets
from keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


train_images = tf.image.resize(train_images, [75, 75])
test_images = tf.image.resize(test_images, [75, 75])

train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images[:100]
train_labels = train_labels[:100]

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)

history = model.fit(train_datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) // 32,
                    epochs=10,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')


#Drive video: https://drive.google.com/drive/folders/1MfXoSq1WirXE7OUZqI29bjwyVKzc2H-n?usp=sharing