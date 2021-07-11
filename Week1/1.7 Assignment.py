import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, MaxPooling2D)


def scale_mnist_data(train_images, test_images):
    """
    This function takes in the training and test images as loaded in the cell above, and scales them
    so that they have minimum and maximum values equal to 0 and 1 respectively.
    Your function should return a tuple (train_images, test_images) of scaled training and test images.
    """
    return train_images / 255., test_images / 255.


def get_model(input_shape):
    """
    This function should build a Sequential model according to the above specification. Ensure the
    weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument.
    Your function should return the model.
    """
    model = Sequential([
        Conv2D(8, (3, 3), padding='SAME', activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


def compile_model(model):
    """
    This function takes in the model returned from your get_model function, and compiles it with an optimiser,
    loss function and metric.
    Compile the model using the Adam optimiser (with default settings), the cross-entropy loss function and
    accuracy as the only metric.
    Your function doesn't need to return anything; the model will be compiled in-place.
    """
    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def train_model(model, scaled_train_images, train_labels):
    """
    This function should train the model for 5 epochs on the scaled_train_images and train_labels.
    Your function should return the training history, as returned by model.fit.
    """
    return model.fit(scaled_train_images, train_labels, epochs=5)

def evaluate_model(model, scaled_test_images, test_labels):
    """
    This function should evaluate the model on the scaled_test_images and test_labels.
    Your function should return a tuple (test_loss, test_accuracy).
    """
    test_loss, test_accuracy = model.evaluate(scaled_test_images, test_labels)
    return test_loss, test_accuracy

if __name__ == '__main__':

    mnist_data = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

    scaled_train_images, scaled_test_images = scale_mnist_data(train_images, test_images)
    scaled_train_images = scaled_train_images[..., np.newaxis]
    scaled_test_images = scaled_test_images[..., np.newaxis]

    model = get_model(scaled_train_images[0].shape)
    compile_model(model)
    history = train_model(model, scaled_train_images, train_labels)

    frame = pd.DataFrame(history.history)
    acc_plot = frame.plot(y="accuracy", title="Accuracy vs Epochs", legend=False)
    acc_plot.set(xlabel="Epochs", ylabel="Accuracy")

    test_loss, test_accuracy = evaluate_model(model, scaled_test_images, test_labels)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

    num_test_images = scaled_test_images.shape[0]

    random_inx = np.random.choice(num_test_images, 4)
    random_test_images = scaled_test_images[random_inx, ...]
    random_test_labels = test_labels[random_inx, ...]

    predictions = model.predict(random_test_images)

    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.4, wspace=-0.2)

    for i, (prediction, image, label) in enumerate(zip(predictions, random_test_images, random_test_labels)):
        axes[i, 0].imshow(np.squeeze(image))
        axes[i, 0].get_xaxis().set_visible(False)
        axes[i, 0].get_yaxis().set_visible(False)
        axes[i, 0].text(10., -1.5, f'Digit {label}')
        axes[i, 1].bar(np.arange(len(prediction)), prediction)
        axes[i, 1].set_xticks(np.arange(len(prediction)))
        axes[i, 1].set_title(f"Categorical distribution. Model prediction: {np.argmax(prediction)}")

    plt.show()