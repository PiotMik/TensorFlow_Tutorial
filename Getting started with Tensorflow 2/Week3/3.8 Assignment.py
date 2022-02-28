from numpy.random import seed
from tensorflow_estimator.python.estimator import early_stopping

from course_utils import plot_compare_metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization)
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)

seed(8)


def read_in_and_split_data(iris_data):
    """
    This function takes the Iris dataset as loaded by sklearn.datasets.load_iris(), and then
    splits so that the training set includes 90% of the full dataset, with the test set
    making up the remaining 10%.
    Your function should return a tuple (train_data, test_data, train_targets, test_targets)
    of appropriately split training and test data and targets.

    If you would like to import any further packages to aid you in this task, please do so in the
    Package Imports cell above.
    """
    data = iris_data['data']
    targets = iris_data['target']
    return train_test_split(data, targets, test_size=0.1)


def get_model(input_shape):
    """
    This function should build a Sequential model according to the above specification. Ensure the
    weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument.
    Your function should return the model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape,
              kernel_initializer=HeUniform, bias_initializer='ones'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    return model


def get_regularised_model(input_shape, dropout_rate, weight_decay):
    """
    This function should build a regularised Sequential model according to the above specification.
    The dropout_rate argument in the function should be used to set the Dropout rate for all Dropout layers.
    L2 kernel regularisation (weight decay) should be added using the weight_decay argument to
    set the weight decay coefficient in all Dense layers that use L2 regularisation.
    Ensure the weights are initialised by providing the input_shape argument in the first layer, given by the
    function argument input_shape.
    Your function should return the model.
    """

    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape,
              kernel_initializer=HeUniform, bias_initializer='ones',
              kernel_regularizer=l2(weight_decay)),
        Dense(128, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dense(128, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(128, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dense(128, activation='relu', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dense(64, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dense(64, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dense(3, activation='softmax')
    ])
    return model


def compile_model(model):
    """
    This function takes in the model returned from your get_model function, and compiles it with an optimiser,
    loss function and metric.
    Compile the model using the Adam optimiser (with learning rate set to 0.0001),
    the categorical crossentropy loss function and accuracy as the only metric.
    Your function doesn't need to return anything; the model will be compiled in-place.
    """
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])


def train_model(model, train_data, train_targets, epochs):
    """
    This function should train the model for the given number of epochs on the
    train_data and train_targets.
    Your function should return the training history, as returned by model.fit.
    """
    history = model.fit(train_data,
                        train_targets,
                        epochs=epochs,
                        batch_size=40,
                        validation_split=0.15)
    return history


def plot_epoch_vs_metric(history,
                         metric_name):
    """
    This function plots the graph of a chosen metric.
    """
    plt.plot(history.history[metric_name])
    plt.plot(history.history[f"val_{metric_name}"])
    plt.title(f'{metric_name.upper()} vs. epochs')
    plt.ylabel(f'{metric_name}')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()


def get_callbacks():
    """
    This function should create and return a tuple (early_stopping, learning_rate_reduction) callbacks.
    The callbacks should be instantiated according to the above requirements.
    """
    early_stopping = EarlyStopping(patience=30,
                                   mode='min')
    learning_rate_reduction = ReduceLROnPlateau(patience=20,
                                                factor=0.2)
    return early_stopping, learning_rate_reduction

# Load and split data
iris_data = load_iris()
train_data, test_data, train_targets, test_targets = read_in_and_split_data(iris_data)

# Refactor labels to one-hot-encoding
train_targets = to_categorical(np.array(train_targets))
test_targets = to_categorical(np.array(test_targets))

# Initialize, compile and train model
model = get_model(train_data[0].shape)
model_regularized = get_regularised_model(train_data[0].shape, 0.3, 0.001)
model_regularized_with_callbacks = get_regularised_model(train_data[0].shape, 0.3, 0.001)

compile_model(model)
compile_model(model_regularized)
compile_model(model_regularized_with_callbacks)

history = train_model(model,
                      train_data, train_targets,
                      epochs=800)
history_regularized = train_model(model_regularized,
                                  train_data, train_targets,
                                  epochs=800)

early_stopping, learning_rate_reduction = get_callbacks()
history_regularized_with_callbacks = model_regularized_with_callbacks.fit(
    train_data, train_targets,
    epochs=800, validation_split=0.15,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=0
)

# Plot epoch vs acc graph
plot_compare_metrics(history, history_regularized, metric_name='acc')
plot_compare_metrics(history, history_regularized, metric_name='loss')

plot_compare_metrics(history_regularized, history_regularized_with_callbacks, metric_name='acc')
plot_compare_metrics(history_regularized, history_regularized_with_callbacks, metric_name='loss')

# Evaluate on test set
test_loss, test_acc = model_regularized_with_callbacks.evaluate(test_data, test_targets, verbose=0)
print(f"""
Test loss: {test_loss:.3f}
Test acc : {100*test_acc:.2f}%""")

