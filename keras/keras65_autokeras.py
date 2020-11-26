# !pip install autokeras
# !pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc4

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak

x=np.load('./data/keras64_x.npy')
y=np.load('./data/keras64_y.npy')

print(x.shape) #(1736, 150, 150, 3)
print(y.shape) #(1736,)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

clf=ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)

clf.fit(x_train, y_train, epochs=50)

predicted_y=clf.predict(x_test)

print('predicted_y :', predicted_y)
print('evaluate() :', clf.evaluate(x_test, y_test))

# evaluate() : [1.8139715194702148, 0.550000011920929]