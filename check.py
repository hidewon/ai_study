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
	# read weighted datas of models for tensorflows.
	model.load_weights('flower.hdf5')
	return model

def check(model,fname):
	# normalize datas after load images.
	img=Image.open(fname)
	img=img.convert('RGB')
	img=img.resize((photo_size,photo_size))
	data=np.asarray(img).reshape((-1,data_size))/ 256

	# predict name of flowers in the images.
	res=model.predict([data])[0]
	y=res.argmax() # answer (max values is answer.)
	per=int(res[y] * 100) # get correct rate.
	print("{0} ({1} %)".format(labels[y], per))

if len(sys.argv) <= 1:
	print('checy.py [filename]')
	quit()

model=build_model()
check(model,sys.argv[1])

