from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Flatten,
                                     Conv2D, MaxPooling2D)

if __name__ == '__main__':
    model = Sequential([
        Conv2D(16,  # how many filters
               (3, 3),  # moving window dimensions
               padding='SAME',  # to keep the same shape as input
               strides=2,  # the strides of the moving window
               activation='relu',
               data_format='channels_first',
               input_shape=(28, 28, 1)),
        MaxPooling2D((3, 3)),  # runs a pooling filter (moving window) on data, and takes max
        Flatten(),
        Dense(10, activation='softmax')

    ])

    model.summary()
