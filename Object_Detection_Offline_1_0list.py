# Import required libraries
import os
import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load the YOLOv5 model
weights = 'V10.pt'
device = select_device('cpu')  # or 'cuda:0' for GPU
model = attempt_load(weights, device)

# Set the model to evaluation mode
model.eval()

# Define the classes
classes = ['Uncovered area', 'Hot spot']
# Create a dictionary to map class names to their corresponding BGR colors
class_colors = {
    'Hot spot': (255, 0, 0),  # Red in RGB
    'Uncovered area': (250, 128, 114),# Salmon in RGB
}

# Function to maintain aspect ratio
def maintain_aspect_ratio(img, target_size, stride=32):
    img_height, img_width = img.shape[:2]
    aspect_ratio = img_width / img_height

    # Check if the image is horizontal or vertical
    if aspect_ratio >= 1:  # Horizontal
        new_width = target_size
        new_height = int(new_width / aspect_ratio)
    else:  # Vertical
        new_height = target_size
        new_width = int(new_height * aspect_ratio)

    # Ensure dimensions are divisible by stride
    new_width = (new_width // stride) * stride
    new_height = (new_height // stride) * stride

    img_resized = cv2.resize(img, (new_width, new_height))

    return img_resized

# Define the confidence threshold and non-maximum suppression threshold
conf_threshold = 0.5
nms_threshold = 0.1

image_list = [
    "GOPR0110.JPG",
    "GOPR0117.JPG",
    "GOPR0125.JPG",
    "IMG_3260.JPG",
    "IMG_3273.JPG",
    "IMG_3282.JPG",
]

gopro_images = None
img_images = None

for image_path in image_list:
    img = cv2.imread(os.path.join('Resources', 'Full_image_set_resized', image_path))

    # Preprocess the input image
    img = maintain_aspect_ratio(img, 640)
    img = img[..., ::-1]  # BGR to RGB
    img = np.ascontiguousarray(img)

    # Check if the current image is a GoPro or IMG image
    if image_path.startswith("GOPR"):
        if gopro_images is None:
            gopro_images = img
        else:
            gopro_images = np.hstack((gopro_images, img))
    elif image_path.startswith("IMG"):
        if img_images is None:
            img_images = img
        else:
            img_images = np.hstack((img_images, img))

# Save the concatenated images
cv2.imwrite("gopro_images.jpg", gopro_images[..., ::-1])
cv2.imwrite("img_images.jpg", img_images[..., ::-1])

# Show the concatenated images
cv2.imshow("GoPro images", gopro_images[..., ::-1])
cv2.imshow("IMG images", img_images[..., ::-1])
cv2.waitKey()
cv2.destroyAllWindows()