# Recognizing posts with YOLOv5
A Streamlit based application that allows character recognition based on the use of the YOLOv5 model. 
The model was trained on a set of images, thanks to which it is equipped with the characteristics of selected characters from the "Captain Bomba" series. 
The application is removable in the aftermath and once deployed can be used online via a link:https://bomba-kapitan.streamlit.app/

## Functionality
* **Loading photos**: The user can have a photo in .jpg, .jpeg, .png format.
* **Character Recognition**: The **YOLOv5 model** is distinctive and marked on the character.
* **Detection results**: The output image has green frames with labels that represent the difference between the recognized character and the detection threat.

## Technology
* **Roboflow**: a framework for preparing datasets
* **Python**: The main language for building applications.
* **Streamlit**: a framework for building data analytics web applications.
* **YOLOv5**: a model for object detection.
* **OpenCV**: an image processing library.
* **Pillow** (PIL): An image manipulation library.
* **PyTorch**: a framework for implementing the YOLOv5 model.
