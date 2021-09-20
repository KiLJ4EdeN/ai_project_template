"""user interface"""
import streamlit as st
from PIL import Image
import numpy as np
from preprocessing import preprocess_image
from models import SimpleCNN, TFLiteModel


# UI
st.write("MNIST digit prediction")

# html input with an extra extension checker
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

model_name = st.sidebar.selectbox(
    'Select Model',
    [None, "keras", "tflite"])

if file is None:
    st.text("Please upload an image file:")
else:
    img = Image.open(file)
    st.image(img, use_column_width=True)
    img = preprocess_image(img)
    img = img.reshape(1, 28, 28, 1)
    # the prediction procedure should not be any different
    if model_name == 'keras':
        simple_cnn = SimpleCNN(model_path='../weights/mnist.h5',
                               input_shape=(28, 28, 1),
                               num_classes=10)
        print(f'[INFO] Predicting with keras...')
        pred = simple_cnn.predict(img)
    elif model_name == 'tflite':
        tflite_cnn = TFLiteModel(model_path='../weights/mnist.tflite')
        print(f'[INFO] Predicting with tflite...')
        pred = tflite_cnn.predict(img)
    else:
        simple_cnn = SimpleCNN(model_path='../weights/mnist.h5',
                               input_shape=(28, 28, 1),
                               num_classes=10)
        # default to keras
        pred = simple_cnn.predict(img)
    label = np.argmax(pred)
    st.write(f'the prediction is: {label}')
    st.write('class probs:')
    st.write([f'{i}: {pred[0][i]}' for i in range(10)])
