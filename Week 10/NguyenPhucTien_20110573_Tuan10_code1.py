#Last Edit: 10/06/2023
#Author: Nguyen Phuc Tien


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications import InceptionV3, VGG16
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.applications.inception_v3 import decode_predictions as inception_decode_predictions
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from keras.applications.vgg16 import decode_predictions as vgg_decode_predictions
from PIL import Image

# Load InceptionV3 model
inception_model = InceptionV3(weights='imagenet', include_top=True)

# Load VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=True)

def predict_image_class(model, preprocess_input, image_array, target_size):
    img = image.array_to_img(image_array)
    img = img.resize(target_size) 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

folder_path = "resized_images"

image_filenames = os.listdir(folder_path)


for filename in image_filenames:
    img_path = os.path.join(folder_path, filename)
    img = Image.open(img_path)
    img_array = np.array(img)
    
    inception_preds = predict_image_class(inception_model, inception_preprocess_input, img_array, (299, 299))
    inception_class_idx = np.argmax(inception_preds)
    inception_class_name = inception_decode_predictions(inception_preds, top=1)[0][0][1]
    inception_proba = inception_preds[0][inception_class_idx]

    vgg_preds = predict_image_class(vgg_model, vgg_preprocess_input, img_array, (224, 224))
    vgg_class_idx = np.argmax(vgg_preds)
    vgg_class_name = vgg_decode_predictions(vgg_preds, top=1)[0][0][1]
    vgg_proba = vgg_preds[0][vgg_class_idx]

    
    fig, ax = plt.subplots(3, 1, figsize=(6, 12))
    
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    
    ax[1].text(0.5, 0.5, f"InceptionV3 Prediction: {inception_class_name} with probability {inception_preds[0][inception_class_idx]}", ha='center', va='center')
    ax[1].axis('off')
    ax[1].set_title('InceptionV3 Prediction')
    
    ax[2].text(0.5, 0.5, f"VGG16 Prediction: {vgg_class_name} with probability {vgg_preds[0][vgg_class_idx]}", ha='center', va='center')
    ax[2].axis('off')
    ax[2].set_title('VGG16 Prediction')
    
    plt.tight_layout()
    plt.show() 

#Drive video: https://drive.google.com/drive/folders/1MfXoSq1WirXE7OUZqI29bjwyVKzc2H-n?usp=sharing