#!/usr/bin/env python3

from keras.modles import Sequential
from keras.layers import Activation, Dense

from PIL import Image

import numpy as np, sys

# declare variables

classes=3
photo_size=75
data_size=photo_size*photo_size*3
lables=["sakura","sunflower","rose"]

# build models

def build_model():
	# define models
	model=Sequential()
	model.add(Dense(units=64,input_dim=(data_size)))
	model.add(Dense(units=classes))
	model.add(Activation('softmax'))
	model.compile(
		loss='sparse_categorical_crossentropy',
		optimizer='sgd',
		metrics=['accuracy']
	)
