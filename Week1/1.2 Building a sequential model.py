from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Flatten, Softmax)

if __name__ == '__main__':
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(16, activation='relu', name = 'Layer1'),
        Dense(16, activation='relu'),
        Dense(10),
        Softmax()
    ])

    print(model.summary())
