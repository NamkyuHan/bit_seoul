import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np
import datetime

num_samples = 100
height = 71
width = 71
num_classes = 100

start1 = datetime.datetime.now()
with tf.device('/gpu:0'):
    model = Xception(weights=None, 
                     input_shape=)



# Generate dummy data.
x = np.




