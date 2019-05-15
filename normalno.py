import random

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop',
              loss='sgd',
              metrics=['accuracy'])
np.expand_dims()
for i in range(20000):
    if i == 100:
        print(1)
        pass
    first = random.random() / 2
    sec = random.random() / 2
    x = np.array([first, sec])
    y = np.array([first + sec])
    model.fit(x, y, epochs=10, batch_size=32)
