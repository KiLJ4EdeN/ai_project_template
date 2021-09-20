"""
models module test
"""
import sys
sys.path.append('../src')
from models import SimpleCNN
import unittest
import numpy as np


# here we can have extra tests with accuracy or other metrics
# to assess the model change when we are doing online training
class TestSimpleCNN(unittest.TestCase):
    """
    this instance tests the simple cnn classification model for the mnist dataset
    """
    def test_simplecnn_predict_sample(self):
        simple_cnn = SimpleCNN(model_path='../weights/mnist.h5',
                               input_shape=(28, 28, 1),
                               num_classes=10)
        dummy_image = np.random.randint(0, 255, size=(1, 28, 28, 1), dtype=np.uint8)
        output = simple_cnn.predict(dummy_image)
        self.assertTrue(output.shape == (1, 10))

    def test_simplecnn_predict_batch(self):
        simple_cnn = SimpleCNN(model_path='../weights/mnist.h5',
                               input_shape=(28, 28, 1),
                               num_classes=10)
        dummy_image = np.random.randint(0, 255, size=(200, 28, 28, 1), dtype=np.uint8)
        output = simple_cnn.predict(dummy_image)
        self.assertTrue(output.shape == (200, 10))
