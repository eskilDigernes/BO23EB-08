import torch

# Models
model_asbjorn = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
model_stian = torch.hub.load('ultralytics/yolov5', 'custom', 'weight_all.pt')

# List of images
image_paths = [

#Full resolution
'Resources\Full_Image_set\IMG_3258.jpg',
'Resources\Full_Image_set\IMG_3273.jpg',
'Resources\Full_Image_set\GOPR0125.jpg',
'Resources\Full_Image_set\IMG_7438.jpg',
'Resources\Full_Image_set\IMG_3282.jpg',
'Resources\Full_Image_set\IMG_3284.jpg',

# 1 mega pixel
'Resources\Full_Image_set_resized\GOPR0117.jpg',
'Resources\Full_Image_set_resized\GOPR0125.jpg',
'Resources\Full_Image_set_resized\IMG_3269.jpg',
'Resources\Full_Image_set_resized\IMG_3273.jpg',
'Resources\Full_Image_set_resized\IMG_7428.jpg',
'Resources\Full_Image_set_resized\IMG_7437.jpg',
'Resources\Full_Image_set_resized\IMG_7438.jpg',
]

# Show results ASBJORN
for img in image_paths:
    results = model_asbjorn(img)
    results.show()

# Show results STIAN
for img in image_paths:
    results = model_stian(img)
    results.show()
