from tensorflow.keras.callbacks import (LearningRateScheduler,
                                        CSVLogger,
                                        LambdaCallback,
                                        ReduceLROnPlateau)
from course_utils import *

model = Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])


# LearningRateScheduler callback
# It allows to tweak the learning rate at learning runtime.
# >>> LearningRateScheduler(schedule, verbose=0)
# where schedule is a function:
# >>> schedule(current_epoch: int,
#              current_learning_rate: float) -> new_learning_rate: float

def lr_schedule(epoch: int, lr: float) -> float:
    if epoch % 2 == 0:
        return lr
    else:
        return lr + epoch/1000.0


history = model.fit(train_data, train_targets, epochs=10,
                    callbacks=[LearningRateScheduler(lr_schedule, verbose=1)],
                    verbose=False)

history2 = model.fit(train_data, train_targets, epochs=10,
                     callbacks=[LearningRateScheduler(lambda x:1/(3+5*x),
                                                      verbose=1)],
                     verbose=False)

# CSV logger callback
# It allows one to store the metrics values computed during training
# and save it in a CSV file.
# >>> CSVLogger(filename: str,
#               separator: str = ',',
#               append: bool = False)
history3 = model.fit(train_data, train_targets, epochs=10,
                     callbacks=[CSVLogger("results.csv")],
                     verbose=False)
pd.read_csv("results.csv", index_col='epoch')


# LambdaCallback
# It allows one to define simple, oneliner custom callbacks
# >>> LambdaCallback(on_epoch_begin = None, on_epoch_end = None,
#                    on_batch_begin = None, on_batch_end = None,
#                    on_train_begin = None, on_train_end = None)
#
# Each parameter of LambdaCallback is a custom function:
# >>> on_epoch_begin(epoch, logs)
# >>> on_epoch_end(epoch, logs)
# >>> on_batch_begin(batch, logs)
# >>> on_batch_end(batch, logs)
# >>> on_train_begin(logs)
# >>> on_train_end(logs)

epoch_callback = LambdaCallback(
    on_epoch_begin=lambda epoch, logs: print(f'Starting Epoch {epoch + 1}')
)
batch_loss_callback = LambdaCallback(
    on_batch_end=lambda batch, logs: print(f"After batch {batch}, the loss is {logs['loss']:7.2f}.")
)
train_finish_callback = LambdaCallback(
    on_train_end=lambda logs: print("Training finished!")
)

history4 = model.fit(train_data, train_targets,
                     epochs=5, batch_size=100,
                     callbacks=[epoch_callback,
                                batch_loss_callback,
                                train_finish_callback],
                     verbose=False)

# ReduceLROnPlateau
# It allows one to automatically tweak the learning rate
# if a metric stops improving.
# >>> ReduceLROnPlateau(
#       monitor='val_loss', # metric to look at
#       factor=0.1,         # new_lr = factor*old_lr
#       patience=10,        # how many epochs to wait until lr change
#       verbose=0,
#       mode='auto',        # ['auto','min','max'] what 'improving' means?
#       min_delta=0.0001,   # threshold for what counts as improvement
#       cooldown=0,         # after lr_change, stop monitoring for a few epochs
#       min_lr=0)           # lower bound on the lr

history5 = model.fit(train_data, train_targets,
                     epochs=100, batch_size=100,
                     callbacks=[ReduceLROnPlateau(
                         monitor='loss',
                         factor=0.2,
                         verbose=1
                     )],
                     verbose=False)
