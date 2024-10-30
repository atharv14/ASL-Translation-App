import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("OpenCV version:", cv2.__version__)

# Test TensorFlow and Keras
model = keras.Sequential([keras.layers.Dense(1, input_shape=(1,))])
print("TensorFlow model created successfully")

# Test OpenCV
cap = cv2.VideoCapture(0)  # Opens the default camera
if cap.isOpened():
    print("OpenCV VideoCapture works")
    ret, frame = cap.read()
    if ret:
        resized_frame = cv2.resize(frame, (224, 224))
        print("OpenCV resize works")
    cap.release()
else:
    print("Failed to open camera. This is expected if running on a system without a camera.")

print("All imports and basic functionalities tested successfully")