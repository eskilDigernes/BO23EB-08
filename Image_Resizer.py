from PIL import Image
import os

input_folder = 'Resources/Full_image_set'
output_folder = 'Resources/Full_image_set_resized'
size = (1225, 816) # new size of the image

for filename in os.listdir(input_folder):
    if filename.startswith('GOP'):
        size = (1225, 816) # landscape size
    elif filename.startswith('IMG'):
        size = (816, 1225) # portrait size
    else:
        continue # skip files that don't match the pattern

    img = Image.open(os.path.join(input_folder, filename))
    w, h = img.size
    aspect_ratio = w / h
    target_ratio = size[0] / size[1]

    if aspect_ratio > target_ratio:
        new_w = size[0]
        new_h = int(size[0] / aspect_ratio)
    else:
        new_w = int(size[1] * aspect_ratio)
        new_h = size[1]

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    img_resized.save(os.path.join(output_folder, filename))