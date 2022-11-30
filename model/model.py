import tenserflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os

### Load data
mnist = keras.datasets.mnist
(train_images, train_images_labels), (test_images, test_images_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images_labels = test_images_labels / 255.0

train_images_labels = np.expand_dims(train_images, axis=3)
test_images_labels = np.expand_dims(test_images, axis=3)