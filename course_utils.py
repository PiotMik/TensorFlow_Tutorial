import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPooling2D, Flatten)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Week 3 utils:


diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']


def get_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    return model


def get_regularized_model(wd, rate):
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(wd),
              input_shape=(train_data.shape[1],)),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(1)
    ])
    return model


def plot_performance(history,
                     metric_name: str = None):
    if metric_name is not None:
        return plot_loss_and_metric(history, metric_name)
    else:
        return plot_loss(history)


def plot_loss(history):
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)

    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    plt.title('Loss vs epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    ax.legend(['Training', 'Validation'], loc='upper right')

    return fig, ax


def plot_loss_and_metric(history, metric_name='mae'):
    frame = pd.DataFrame(history.history)
    epochs = np.arange(len(frame))

    fig = plt.figure(figsize=(12, 4))

    # Loss plot
    ax = fig.add_subplot(121)
    ax.plot(epochs, frame['loss'], label="Train")
    ax.plot(epochs, frame['val_loss'], label="Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Epochs")
    ax.legend()

    # Metric plot
    ax = fig.add_subplot(122)
    ax.plot(epochs, frame[metric_name], label="Train")
    ax.plot(epochs, frame[f'val_{metric_name}'], label="Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(f"{metric_name}")
    ax.set_title(f"{metric_name.upper()} vs Epochs")
    ax.legend()
    return fig, ax


def plot_compare_metrics(history1, history2,
                         metric_name='loss'):
    fig = plt.figure(figsize=(12, 5))
    fig.add_subplot(121)
    plt.plot(history1.history[metric_name])
    plt.plot(history1.history[f"val_{metric_name}"])
    plt.title(f'Model 1: {metric_name} vs Epochs')
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Epochs")
    plt.legend(['Training', 'Validation'], loc='upper right')

    fig.add_subplot(122)
    plt.plot(history2.history[metric_name])
    plt.plot(history2.history[f'val_{metric_name}'])
    plt.title(f'Model 2: {metric_name} vs Epochs')
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Epochs")
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.show()


# Week 4 utils:


train_data, test_data, train_targets, test_targets = train_test_split(data, targets)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]


def get_test_accuracy(model, x, y):
    test_loss, test_acc = model.evaluate(x=x, y=y, verbose=1)
    print(f'Accuracy: {test_acc:0.3f}')


def get_new_model():
    model = Sequential([
        Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3, 3),
               activation='relu', name='conv_1'),
        Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
               name='conv_2'),
        MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32, activation='relu', name='dense_1'),
        Dense(units=10, activation='relu', name='dense_2')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def generate_checkpoint_path(filename: str = "checkpoint"):
    base = os.getcwd()
    now = datetime.now()
    date_part = f"{now:%Y%m%d}"
    time_part = f"{now:%I_%M_%S_%p}"
    full_path = os.path.join(base, 'model_checkpoints', date_part, time_part)

    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return os.path.join(full_path, filename)
