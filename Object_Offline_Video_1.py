
import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load the YOLOv5 model
weights = 'weight_all.pt'
device = select_device('cpu')  # or 'cuda:0' for GPU
model = attempt_load(weights, device)

# Set the model to evaluation mode
model.eval()

# Define the classes
classes = ['Red spot', 'Black spot', 'Nipple', 'Extraction point', 'Point flame']

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

# Open the input video
input_video_path = 'Resources\input_video.mp4'
cap = cv2.VideoCapture(input_video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print('Error: Could not open the video file.')
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up the output video writer
output_video_path = 'Resources/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 video format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


# Define the class colors
class_colors = {
    'Red spot': (255, 0, 0),  # Red in RGB
    'Black spot': (250, 128, 114),  # Salmon in RGB
    'Nipple': (245, 197, 66),  # Orange in RGB
    'Point flame': (182, 48, 209),  # Purple in RGB
    'Extraction point': (0, 255, 0),  # Green in RGB
}

# Process the input video frame by frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    img = maintain_aspect_ratio(frame, 640)
    img = img[..., ::-1]  # BGR to RGB
    img = np.ascontiguousarray(img)

    # Convert the frame to a tensor
    img_vis = img.copy()  # Create a copy of the original frame for visualization
    img = img.transpose((2, 0, 1))  # Move channels to the first dimension
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run object detection on the frame
    outputs = model(img)
    results = non_max_suppression(outputs, conf_threshold, nms_threshold)

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

    # Write the visualized frame to the output video
    out.write(img_vis[..., ::-1])

    # Show the output frame
    cv2.imshow('output', img_vis[..., ::-1])

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects and close the display window
# cap.release()
# out.release()
# cv2.destroyAllWindows()
