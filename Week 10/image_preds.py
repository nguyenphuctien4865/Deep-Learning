import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, target_size=(224, 224), image_format='JPEG'):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Open the image using PIL
            img = Image.open(input_path)

            # Resize the image
            img = img.resize(target_size)

            # Save the resized image
            img.save(output_path)

            print(f"Resized and saved {filename} successfully.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Example usage
input_folder = "image"
output_folder = "resized_images"
resize_images_in_folder(input_folder, output_folder, target_size=(224, 224))