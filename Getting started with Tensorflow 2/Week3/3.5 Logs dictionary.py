# Logs dictionary
# When working with callbacks, an argument is being passed after each batch/epoch.
# This argument is "logs" and contains information on loss + metrics defined upon model compilation.
# You can use this information in various ways, from simply informing user by printing out a chosen metric,
# up to even early stopping of training, when a certain condition involving metrics is met."""

from course_utils import *
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import Callback

model = Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])


class LossAndMetricCallback(Callback):

    # Print loss after every second batch in training
    def on_train_batch_end(self, batch, logs=None):
        if batch % 2 == 0:
            print(f"After batch {batch}, the loss is {logs['loss']:.2f}")

    # Print loss after each batch in test (validation)
    def on_test_batch_end(self, batch, logs=None):
        print(f"After batch {batch}, the loss is {logs['loss']:.2f}")

    # Print loss and MAE after each epoch
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: Average loss is {logs['loss']:.2f}, mean absolute error is {logs['mae']:.2f}")

    # Notify at the end of batch
    def on_predict_batch_end(self, batch, logs=None):
        print(f"Finished prediction on batch {batch}!")


history = model.fit(train_data, train_targets, epochs=20,
                    batch_size=100, callbacks=[LossAndMetricCallback()], verbose=False)

model_eval = model.evaluate(test_data, test_targets, batch_size=10,
                            callbacks=[LossAndMetricCallback()], verbose=False)

model_prediction = model.predict(test_data, batch_size=10,
                                 callbacks=[LossAndMetricCallback()], verbose=False)

# Example of use: learning_rate scheduler
# Let's tweak the learning rate at model's runtime

lr_schedule = {
    4: 0.03,
    7: 0.02,
    11: 0.005,
    15: 0.007
}


def get_new_epoch_lr(epoch, lr):
    """Check to see if the input epoch is listed in the learning rate schedule.
    If it is, return the assigned learning rate"""

    if epoch in lr_schedule:
        return lr_schedule[epoch]
    else:
        return lr


class LRScheduler(Callback):

    def __init__(self, new_lr):
        super(LRScheduler, self).__init__()
        self.new_lr = new_lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError("Optimizer does not have a learning rate parameter.")

        curr_rate = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        scheduled_rate = self.new_lr(epoch, curr_rate)

        tf.keras.backend.set_value(self.model.optimizer.lr,
                                   scheduled_rate)
        print(f"Learning rate for epoch {epoch} is {scheduled_rate:7.3f}")


new_model = Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

new_model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])

new_history = new_model.fit(train_data, train_targets, epochs=20,
                            batch_size=100, callbacks=[LRScheduler(get_new_epoch_lr)], verbose=False)
