

from __future__ import print_function

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import os
import pandas as pd
import random
img_rows=256
img_cols=256
x=np.random.rand(2,4)
print(x.shape)
print(x)
pred=np.zeros([2,1])
for i in range(2):
    pred[i,0]=np.argmax(x[i,:])
print(pred)