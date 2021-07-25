# Saving and loading model weights
# TensorFlow allows to save model parameters,
# or even whole models at runtime using callbacks

from course_utils import *
from tensorflow.keras.callbacks import ModelCheckpoint

# Visualize our data/images:
fig, ax = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    ax[i].set_axis_off()
    ax[i].imshow(x_train[i])


def get_test_accuracy(model_, x_test_, y_test_):
    test_loss, test_acc = model_.evaluate(x=x_test_, y=y_test_,
                                          verbose=1)
    print(f'Accuracy: {test_acc:0.3f}')


def get_new_model():
    model_ = Sequential([
        Conv2D(filters=16, input_shape=(32, 32, 1), kernel_size=(3, 3),
               activation='relu', name='conv_1'),
        Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
               name='conv_2'),
        MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32, activation='relu', name='dense_1'),
        Dense(units=10, activation='relu', name='dense_2')
    ])
    model_.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return model_


model = get_new_model()
model.summary()
get_test_accuracy(model, x_test, y_test)

# ModelCheckpoint
# This callback will save your model in a way that's
# dependent on specification.
# ModelCheckpoint(filepath: str,
#                 save_weights_only: bool)
#
# filepath argument can simply specify a base-name,
# like 'checkpoint'. Then the outputs will be 3 files:
# >>> checkpoint
# >>> checkpoint.data-00000-of-00001
# >>> checkpoint.index
#
# Otherwise, if filename ends with '.h5'
# (native 'HDF' TensorFlow format), only 1 file will be saved.
# >>> checkpoint.h5

checkpoint_path = generate_checkpoint_path('checkpoint')
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_freq='epoch',
                             save_weights_only=True,
                             verbose=1)

# Train the model and save weights
model.fit(x=x_train, y=y_train,
          epochs=3, callbacks=[checkpoint])

# Get a new, untrained model and load trained weights
model = get_new_model()
get_test_accuracy(model, x_test, y_test)
model.load_weights(checkpoint_path)
get_test_accuracy(model, x_test, y_test)
