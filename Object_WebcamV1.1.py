import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Load the YOLOv5 model
weights_choice = int(input("Enter 10 for V10, 20 for V20 and 21 for V21: "))
if weights_choice == 10:
    weights = 'V10.pt'
elif weights_choice == 20:
    weights = 'V20.pt'
elif weights_choice == 21:
    weights = 'V21.pt'
else:
    print("Invalid input. Please try again.")
    exit()

device = select_device('cuda:0')

model = attempt_load(weights, device)
model.eval()

# Define the classes
if weights == 'V21.pt':
    classes = ['Hot spot', 'Point flame', 'Stub', 'Tapping hole', 'Uncovered area']
elif weights == 'V20.pt':
    classes = ['Hot spot', 'Uncovered area', 'Stub', 'Tapping hole', 'Point flame']
elif weights == 'V10.pt':
    classes = ['Uncovered area', 'Hot spot']

def maintain_aspect_ratio(img, target_size):
    img_height, img_width = img.shape[:2]
    aspect_ratio = img_width / img_height

    if img_width >= img_height:
        new_width = target_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(new_height * aspect_ratio)

    new_width = (new_width // 32) * 32
    new_height = (new_height // 32) * 32

    img_resized = cv2.resize(img, (new_width, new_height))
    return img_resized

if weights == 'V21.pt':
    conf_threshold = 0.4
    nms_threshold = 0.6
elif weights == 'V20.pt':
    conf_threshold = 0.6
    nms_threshold = 0.5
elif weights == 'V10.pt':
    conf_threshold = 0.4
    nms_threshold = 0.5

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Error: Could not open the video file.')
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_video_path = 'Resources/output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

class_colors = {
    'Hot spot': (255, 0, 0),
    'Uncovered area': (250, 128, 114),
    'Stub': (245, 197, 66),
    'Point flame': (182, 48, 209),
    'Tapping hole': (0, 255, 0),
}

cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = maintain_aspect_ratio(frame, 1080)
    img = img[..., ::-1]
    img = np.ascontiguousarray(img)

    img_vis = img.copy()
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    outputs = model(img)
    results = non_max_suppression(outputs, conf_threshold, nms_threshold)

    for result in results:
        if result is not None:
            result[:, :4] = scale_coords(img.shape[2:], result[:, :4], img.shape[2:]).round()
            for x1, y1, x2, y2, conf, cls in result:
                label = classes[int(cls)]
                conf_percent = conf * 100
                print(f'{label}: {conf_percent:.1f}%')
                color = class_colors[label]
                cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                (text_width, text_height), _ = cv2.getTextSize(f'{label}: {conf_percent:.1f}%',
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                cv2.rectangle(img_vis, (int(x1), int(y1) - text_height), (int(x1) + text_width, int(y1)),
                              color, -1)

                cv2.putText(img_vis, f'{label}: {conf_percent:.1f}%', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (255, 255, 255), 1, cv2.LINE_AA)

    out.write(cv2.resize(img_vis[..., ::-1], (frame_width, frame_height)))

    cv2.imshow('output', img_vis[..., ::-1])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()

cv2.destroyAllWindows()
