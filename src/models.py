"""Models are to be implemented here, or if too long in separate files
The obligation that this creates is that you have to include the methods from the base class in your model.
"""
from base_classes import ClassificationModel
from typing import Tuple
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np


class SimpleCNN(ClassificationModel):
    """
    A simple Convolutional Neural Network
    """
    def __init__(self, model_path: str,
                 input_shape: Tuple[int, int, int] = (28, 28, 1),
                 num_classes: int = 10,):
        """
        :param model_path: where the model is located
        :param input_shape: input shape for the model to be built with
        :param num_classes: number of classes in the classification problem
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.__load_model()

    def __load_model(self):
        """ model loader """
        model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        model.load_weights(self.model_path)
        return model

    def predict(self, images):
        """
        Use the loaded model to make an estimation.
        :param images: photo to make the prediction on.
        :return: predicted class.
        """
        return self.model.predict(images)


# this is an example of how different models are interpreted in various frameworks
# so we need to have uniform methods for them so that they can work alongside each other
class TFLiteModel(ClassificationModel):
    """
    Reduced Tensorflow Lite Detection Model.
    """

    def __init__(self, model_path: str):
        """
        Initiate the model session.
        """
        self.interpreter = self.__load_model(model_path)

    def __load_model(self, model_path: str) -> tf.lite.Interpreter:
        """
        load the tflite model.
        :param model_path: where to load the model from.
        """
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = interpreter.get_input_details()
        # print(self.input_details)
        self.output_details = interpreter.get_output_details()
        # print(self.output_details)
        # Test the model on random input data.
        input_shape = self.input_details[0]['shape']
        # print(f'input shape is: {input_shape}')
        return interpreter

    def predict(self, images):
        """
        Use the loaded model to make an estimation.
        :param images: photo to make the prediction on.
        :return: predicted class.
        """
        images = np.array(images, dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], images)
        self.interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data


# this is for testing only ofc this file wont be run as main ever.
if __name__ == '__main__':
    ob = SimpleCNN(model_path='../weights/mnist.h5',
                   input_shape=(28, 28, 1),
                   num_classes=10)
