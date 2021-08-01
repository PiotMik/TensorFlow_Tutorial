import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import pandas as pd


def load_eurosat_data():
    data_dir = 'data/'
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    x_test  = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test  = np.load(os.path.join(data_dir, 'y_test.npy'))
    return (x_train, y_train), (x_test, y_test)


def get_new_model(input_shape):

    model = Sequential([
        Conv2D(filters=16,
               input_shape=input_shape,
               kernel_size=3,
               activation='relu',
               padding='SAME',
               name='conv_1'),
        Conv2D(filters=8,
               kernel_size=(3, 3),
               activation='relu',
               padding='SAME',
               name='conv_2'),
        MaxPooling2D(pool_size=(8, 8),
                     name='pool_1'),
        Flatten(name='flatten'),
        Dense(32, activation='relu',
              name= 'dense_1'),
        Dense(10, activation='softmax',
              name='dense_2')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_checkpoint_every_epoch():
    checkpoint = ModelCheckpoint(save_freq='epoch',
                                 save_weights_only=True,
                                 filepath='checkpoints_every_epoch/checkpoint_{epoch:03d}')
    return checkpoint


def get_test_accuracy(model, x_test, y_test):
    """Test model classification accuracy"""
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))


def get_checkpoint_best_only():
    checkpoint = ModelCheckpoint(save_weights_only=True,
                                 save_best_only=True,
                                 monitor='test_accuracy',
                                 filepath='checkpoints_best_only/checkpoint')
    return checkpoint


def get_early_stopping():
    callback = EarlyStopping(patience=3,
                             monitor='accuracy')
    return callback


def get_checkpoint_every_epoch():
    """
    This function should return a ModelCheckpoint object that:
    - saves the weights only at the end of every epoch
    - saves into a directory called 'checkpoints_every_epoch' inside the current working directory
    - generates filenames in that directory like 'checkpoint_XXX' where
      XXX is the epoch number formatted to have three digits, e.g. 001, 002, 003, etc.
    """
    checkpoint = ModelCheckpoint(save_freq='epoch',
                                 save_weights_only=True,
                                 filepath='checkpoints_every_epoch/checkpoint_{epoch:03d}',
                                 verbose=1)
    return checkpoint


def get_checkpoint_best_only():
    """
    This function should return a ModelCheckpoint object that:
    - saves only the weights that generate the highest validation (testing) accuracy
    - saves into a directory called 'checkpoints_best_only' inside the current working directory
    - generates a file called 'checkpoints_best_only/checkpoint'
    """
    checkpoint = ModelCheckpoint(save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_accuracy',
                                 filepath='checkpoints_best_only/checkpoint',
                                 verbose=1)
    return checkpoint


def get_early_stopping():
    """
    This function should return an EarlyStopping callback that stops training when
    the validation (testing) accuracy has not improved in the last 3 epochs.
    HINT: use the EarlyStopping callback with the correct 'monitor' and 'patience'
    """
    callback = EarlyStopping(patience=3,
                             monitor='accuracy',
                             verbose=1)
    return callback


def get_model_last_epoch(model):
    """
    This function should create a new instance of the CNN you created earlier,
    load on the weights from the last training epoch, and return this model.
    """
    filename = tf.train.latest_checkpoint(checkpoint_dir='checkpoints_every_epoch')
    model.load_weights(filename)
    return model


def get_model_best_epoch(model):
    """
    This function should create a new instance of the CNN you created earlier, load
    on the weights leading to the highest validation accuracy, and return this model.
    """
    filename = tf.train.latest_checkpoint(checkpoint_dir='checkpoints_best_only')
    model.load_weights(filename)
    return model


def get_model_eurosatnet():
    """
    This function should return the pretrained EuroSatNet.h5 model.
    """
    model = tf.keras.models.load_model('models/EuroSatNet.h5')
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_eurosat_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = get_new_model(x_train[0].shape)
    model.summary()
    get_test_accuracy(model, x_test, y_test)

    checkpoint_every_epoch = get_checkpoint_every_epoch()
    checkpoint_best_only = get_checkpoint_best_only()
    early_stopping = get_early_stopping()

    callbacks = [checkpoint_every_epoch, checkpoint_best_only, early_stopping]
    model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=callbacks)

    model_last_epoch = get_model_last_epoch(get_new_model(x_train[0].shape))
    model_best_epoch = get_model_best_epoch(get_new_model(x_train[0].shape))
    print('Model with last epoch weights:')
    get_test_accuracy(model_last_epoch, x_test, y_test)
    print('')
    print('Model with best epoch weights:')
    get_test_accuracy(model_best_epoch, x_test, y_test)

    model_eurosatnet = get_model_eurosatnet()
    model_eurosatnet.summary()
    get_test_accuracy(model_eurosatnet, x_test, y_test)

