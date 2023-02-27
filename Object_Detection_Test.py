import torch
from PIL import Image
import glob
from itertools import islice

# Models
model_asbjorn = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
model_stian = torch.hub.load('ultralytics/yolov5', 'custom', 'weight_all.pt')

    
image_list = []
for filename in glob.glob('Resources\Full_Image_set/*.jpg'): #assuming jpg
    im=Image.open(filename)
    image_list.append(im)
    
# Model
model_asbjorn = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
model_stian = torch.hub.load('ultralytics/yolov5', 'custom', 'weight_all.pt')

iterator = islice(image_list, 3)
for img in iterator:
    results = model_asbjorn(img)
    results.show()
    results.save('Resources\Results')   


