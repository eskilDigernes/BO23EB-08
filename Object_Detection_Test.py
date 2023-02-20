import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
                                                       

# Images
img = r"C:\Users\Asbjo\Documents\GitHub\BO23EB-08\Resources\Full_Image_set\IMG_3258.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.