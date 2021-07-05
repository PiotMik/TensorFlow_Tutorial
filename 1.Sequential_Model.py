from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Flatten)

if __name__ == '__main__':
    # Initialization of model
    model1 = Sequential(layers=[Dense(64, activation='relu'),
                                Dense(10, activation='softmax')
                                ])

    # The same as above,
    model2 = Sequential()
    model2.add(Dense(64, activation='relu'))
    model2.add(Dense(10, activation='softmax'))

    # You can tell at the beginning what is the data dimension
    model3 = Sequential(layers=[Dense(64, activation='relu', input_shape=(784,)),
                                Dense(10, activation='softmax')
                                ])

    # Flatten dimensions before sending to the first layer
    model4 = Sequential([Flatten(input_shape= (28, 28)),
                         Dense(64, activation='relu'),
                         Dense(10, aactivation='softmax')])
