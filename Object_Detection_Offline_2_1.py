import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load the YOLOv5 model
weights = 'weight_all.pt'
device = select_device('cpu') # or 'cuda:0' for GPU
model = attempt_load(weights, device)

# Set the model to evaluation mode
model.eval()

# Define the classes
classes = ['Hot spot','Point flame','Stub', 'Tapping hole','Uncovered area']

# Define the input image size
def maintain_aspect_ratio(img, target_size):
    img_height, img_width = img.shape[:2]
    aspect_ratio = img_width / img_height

    if img_width >= img_height:
        new_width = target_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(new_height * aspect_ratio)

    img_resized = cv2.resize(img, (new_width, new_height))
    return img_resized

# Define the confidence threshold and non-maximum suppression threshold
conf_threshold = 0.4
nms_threshold = 0.5

# Load the input image
img = cv2.imread('Resources\Full_image_set_resized\IMG_3245.jpg')

# Preprocess the input image
img = maintain_aspect_ratio(img, 640)
img = img[..., ::-1]  # BGR to RGB
img = np.ascontiguousarray(img)

# Convert the input image to a tensor
img_vis = img.copy()  # Create a copy of the original image for visualization
img = img.transpose((2, 0, 1))  # Move channels to the first dimension
img = torch.from_numpy(img).to(device)
img = img.float()
img /= 255.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

# Run object detection on the input image
outputs = model(img)
results = non_max_suppression(outputs, conf_threshold, nms_threshold)

# Create a dictionary to map class names to their corresponding BGR colors
class_colors = {
    'Hot spot': (255, 0, 0),  # Red in RGB
    'Uncovered area': (250, 128, 114), # Salmon in RGB
    'Stub': (245, 197, 66),  # Orange in RGB
    'Point flame': (182, 48, 209),  # Purple in RGB
    'Tapping hole': (0, 255, 0),  # Green in RGB
}

# Visualize the results
for result in results:
    if result is not None:
        result[:, :4] = scale_coords(img.shape[2:], result[:, :4], img.shape[2:]).round()
        for x1, y1, x2, y2, conf, cls in result:
            label = classes[int(cls)]
            print(f'{label}: {conf:.2f}')
            color = class_colors[label]  # Get the color for the current class label
            cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(img_vis, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)

# Show the output image
cv2.imshow('output', img_vis[...,::-1])

cv2.waitKey()
cv2.destroyAllWindows()
