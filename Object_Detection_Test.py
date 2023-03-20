import torch
from PIL import Image
import glob
from itertools import islice

# Models
model_asbjorn = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
model_stian = torch.hub.load('ultralytics/yolov5', 'custom', 'weight_all.pt')

    
image_list = []
for filename in glob.glob('Resources\Full_image_set_resized/*.jpg'): #assuming jpg
    im=Image.open(filename)
    image_list.append(im)
    
# Model
model_asbjorn = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
model_stian = torch.hub.load('ultralytics/yolov5', 'custom', 'weight_all.pt')

iterator = islice(image_list, len(image_list))
for img in iterator:
    results_asbjorn = model_asbjorn(img)
    results_stian = model_stian(img)
    results_asbjorn.show()
    results_stian.show()
    results_asbjorn.save('Resources\Results')
    results_stian.save('Resources\Results')


