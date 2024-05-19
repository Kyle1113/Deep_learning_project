from PIL import Image
import os

# Define the directory containing the images
image_directory = "./data/image"
# Define the target size
target_size = (800, 800)

# Function to resize images
def resize_images(image_dir, target_size):
    # List all files in the directory
    files = os.listdir(image_dir)
    
    for file in files:
        # Construct the full file path
        file_path = os.path.join(image_dir, file)
        
        try:
            # Open the image file
            with Image.open(file_path) as img:
                # Resize the image
                img_resized = img.resize(target_size)
                # Save the resized image, overwriting the original
                img_resized.save(file_path)
                print(f"Resized and saved: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Perform the resizing
resize_images(image_directory, target_size)
