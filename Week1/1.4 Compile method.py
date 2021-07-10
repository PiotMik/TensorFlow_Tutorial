import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Flatten,
                                     Conv2D, MaxPooling2D)

if __name__ == '__main__':
    model = Sequential([
        Conv2D(16,
               (3, 3),
               padding='SAME',
               strides=2,
               activation='relu',
               data_format='channels_first',
               input_shape=(28, 28, 1)),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(10, activation='softmax')

    ])

    # Parameters can be specified by name (then default specs are used)
    model.compile(optimizer='adam',  # other options: sgd, rmsprop, adam, adadelta, ...
                  loss='sparse_categorical_crossentropy',  #
                  metrics=['accuracy', 'mae']),  # mse, rmse, ...
    print(model.loss)
    print(model.optimizer)
    print(model.metrics)

    # Parameters can be specified as tf.keras objects (more flexibility)
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    mae = tf.keras.metrics.MeanAbsoluteError()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=[acc, mae])
    print(model.loss)
    print(model.optimizer)
    print(model.optimizer.lr)
    print(model.metrics)


