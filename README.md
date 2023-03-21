# YOLOv5 Object Detection for Custom Classes

This is a simple Python script using YOLOv5 to perform object detection on custom classes. The script is designed to detect the following object classes in an input image:

1. Red spot
2. Black spot
3. Nipple
4. Extraction point
5. Point flame

The script uses PyTorch, OpenCV, and NumPy libraries for processing and visualisation.

## Prerequisites

To run the script, you need to have the following libraries installed:

1. PyTorch
2. OpenCV
3. NumPy

You can install them using pip:
pip install torch torchvision opencv-python numpy

## Usage

1. Download the pre-trained YOLOv5 model weights file `weight_all.pt` and place it in the same directory as the script.
2. Place an input image (e.g., `IMG_3245.jpg`) in the `Resources\Full_image_set_resized` directory.
3. Run the script:

python your_script_name.py

The script will process the input image, detect the objects, and display the results with bounding boxes and class labels.

## Customisation

You can modify the following parameters in the script to suit your needs:

1. `weights`: Path to the pre-trained YOLOv5 model weights file.
2. `device`: Set to `'cpu'` to use the CPU, or `'cuda:0'` to use the GPU.
3. `classes`: List of class names to detect.
4. `conf_threshold`: Confidence threshold for object detection (default is 0.4).
5. `nms_threshold`: Non-maximum suppression threshold for merging overlapping bounding boxes (default is 0.5).

## Troubleshooting

If you encounter any issues while running the script, please ensure that:

1. All required libraries are installed correctly.
2. The pre-trained YOLOv5 model weights file (`weight_all.pt`) is in the same directory as the script.
3. The input image is placed in the `Resources\Full_image_set_resized` directory.
4. The input image file path is correct and the image file is not corrupted.

If you still have issues, consider checking the following resources for further assistance:

1. [YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5): The official repository for YOLOv5, which contains the latest updates, documentation, and examples.
2. [PyTorch forum](https://discuss.pytorch.org/): A community-driven forum for discussing and resolving issues related to PyTorch.
3. [OpenCV forum](https://forum.opencv.org/): A community-driven forum for discussing and resolving issues related to OpenCV.
4. [NumPy mailing list](https://numpy.org/community/): A mailing list for NumPy users
