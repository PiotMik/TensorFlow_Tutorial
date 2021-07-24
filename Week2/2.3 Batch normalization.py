# Batch Normalization
# It is conceptually a re-normalization of activations between layer outputs.
# Improves training speed and stability of the neural network.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Flatten, Dense,
                                     Conv2D, MaxPooling2D,
                                     BatchNormalization,
                                     Dropout)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']
targets = (targets - targets.mean(axis=0))/targets.std(axis=0)

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)


model = Sequential([
    Dense(64, input_shape=[train_data.shape[1],], activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
])
model.summary()

# with specific params
model.add(tf.keras.layers.BatchNormalization(
    momentum=0.95,  # reusing previous running mean with some weight
    epsilon=0.005,  # controls numerical stability
    axis=-1,
    # beta and gamma perform addititonal affine transformation
    # by default beta=0, gamma=1
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
    gamma_initializer=tf.keras.initializers.Constant(value=0.9)
))
model.add(Dense(1))  # output layer

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

history = model.fit(train_data, train_targets,
                    epochs=100, validation_split=0.15,
                    batch_size=64, verbose=False)

frame = pd.DataFrame(history.history)
epochs = np.arange(len(frame))

fig = plt.figure(figsize=(12,4))

# Loss plot
ax = fig.add_subplot(121)
ax.plot(epochs, frame['loss'], label="Train")
ax.plot(epochs, frame['val_loss'], label="Validation")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Loss vs Epochs")
ax.legend()

# Accuracy plot
ax = fig.add_subplot(122)
ax.plot(epochs, frame['mae'], label="Train")
ax.plot(epochs, frame['val_mae'], label="Validation")
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Mean Absolute Error vs Epochs")
ax.legend()