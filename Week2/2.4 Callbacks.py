# Callbacks
# They allow you to perform actions in the runtime of
# training, validating or predicting.
# The Callback class has methods like
# 'on_test_epoch_begin', or 'on_predict_batch_end'
# which allow you to insert functions in the middle of the runtime
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Dense, Dropout)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']
targets = (targets - targets.mean(axis=0)) / targets.std(axis=0)

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

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


class TrainingCallback(Callback):

    def on_train_begin(self, logs=None):
        print("Starting training ...")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")

    def on_train_batch_begin(self, batch, logs=None):
        print(f"Traning: Starting batch {batch}")

    def on_train_batch_end(self, batch, logs=None):
        print(f"Training: Finished batch {batch}")

    def on_train_epoch_end(self, epoch, logs=None):
        print(f"Training: Finished epoch {epoch}")

class TestingCallback(Callback):

    def on_test_begin(self, logs=None):
        print("Starting training ...")

    def on_test_batch_begin(self, batch, logs=None):
        print(f"Traning: Starting batch {batch}")

    def on_test_batch_end(self, batch, logs=None):
        print(f"Training: Finished batch {batch}")

    def on_test_epoch_end(self, epoch, logs=None):
        print(f"Training: Finished epoch {epoch}")

class PredictionCallback(Callback):

    def on_predict_begin(self, logs=None):
        print("Starting training ...")

    def on_predict_batch_begin(self, batch, logs=None):
        print(f"Traning: Starting batch {batch}")

    def on_predict_batch_end(self, batch, logs=None):
        print(f"Training: Finished batch {batch}")

    def on_predict_epoch_end(self, epoch, logs=None):
        print(f"Training: Finished epoch {epoch}")

model = get_regularized_model(1e-5, 0.3)
model.compile(optimizer='adam',
              loss='mse')
model.fit(train_data, train_targets,
          epochs=3, batch_size=128,
          verbose=False, callbacks=[TrainingCallback()])

model.evaluate(test_data, test_targets,
               verbose=False, callbacks=[TestingCallback()])

model.predict(test_data, verbose=False,
              callbacks=[PredictionCallback()])