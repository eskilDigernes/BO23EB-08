import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load the YOLOv5 model
weights = 'best.pt'
device = select_device('cpu') # or 'cuda:0' for GPU
model = attempt_load(weights, map_location=device)

# Set the model to evaluation mode
model.eval()

# Define the classes
classes = ['Redspot', 'Blackspot']

# Define the input image size
img_size = (640, 640)

# Define the confidence threshold and non-maximum suppression threshold
conf_threshold = 0.4
nms_threshold = 0.5

# Load the input image
img = cv2.imread('Resources\Full_image_set_resized\IMG_3273.jpg')

# Preprocess the input image
img = cv2.resize(img, img_size)
img = img[...,::-1]  # BGR to RGB
img = np.ascontiguousarray(img)

# Convert the input image to a PyTorch tensor
img = torch.from_numpy(img).to(device)
img = img.float()
img /= 255.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

# Run object detection on the input image
outputs = model(img)
results = non_max_suppression(outputs, conf_threshold, nms_threshold)

# Visualize the results
for result in results:
    if result is not None:
        result[:, :4] = scale_coords(img.shape[2:], result[:, :4], img.shape[2:]).round()
        for x1, y1, x2, y2, conf, cls in result:
            label = classes[int(cls)]
            print(f'{label}: {conf:.2f}')
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output image
cv2.imshow('output', img[...,::-1])
cv2.waitKey()
cv2.destroyAllWindows()
