from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, Flatten, MaxPooling2D)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

if __name__ == '__main__':

    fashion_mnist_data = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((3, 3)),  # runs a pooling filter (moving window) on data, and takes max
        Flatten(),
        Dense(10, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate= 0.005)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    mae = tf.keras.metrics.MeanAbsoluteError()
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=[acc, mae])

    labels = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    ]

    train_images = train_images/255.
    test_images = test_images/255.

    test_loss, test_accuracy, test_mae = model.evaluate(test_images[...,
                                                                    np.newaxis],
                                                        test_labels,
                                                        verbose=2)
    random_inx = np.random.choice(test_images.shape[0])
    inx = 20
    test_image = test_images[inx]
    plt.imshow(test_image)
    plt.show()
    print(f"Label: {labels[test_labels[inx]]}")

    predictions = model.predict(test_image[np.newaxis, ..., np.newaxis])
    print(f"Model Prediction: {labels[np.argmax(predictions)]}")