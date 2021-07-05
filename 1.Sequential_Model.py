from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

if __name__ == '__main__':
    model = Sequential(layers=[Dense(64, activation='relu'),
                               Dense(10, activation='softmax')
                               ])
