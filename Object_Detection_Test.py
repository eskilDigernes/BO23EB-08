import torch

# Model
model_asbjorn = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
model_stian = torch.hub.load('ultralytics/yolov5', 'custom', 'weight_all.pt')

# Images
image_paths = [
    'Resources\Full_Image_set\IMG_3258.jpg',
    'Resources\Full_Image_set\IMG_3273.jpg',
    'Resources\Full_Image_set\GOPR0125.jpg',
    'Resources\Full_Image_set\IMG_7438.jpg'
]

# Show results ASBJORN
# for img in image_paths:
#     results = model_asbjorn(img)
#     results.show()

# Show results STIAN
for img in image_paths:
    results = model_stian(img)
    results.show()

