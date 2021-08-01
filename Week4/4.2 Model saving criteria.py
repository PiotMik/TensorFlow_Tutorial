# You can control model saving with callbacks.
# Below: how to save only the 'best weights'
from course_utils import *
from tensorflow.keras.callbacks import ModelCheckpoint

# Saving after each 5000 batches
checkpoint_5000_path = generate_checkpoint_path('checkpoint_{epoch:02d}')
checkpoint_5000 = ModelCheckpoint(filepath=checkpoint_5000_path,
                                  save_weights_only=True,
                                  save_freq=5000,
                                  verbose=1)
model = get_new_model()
model.fit(x=x_train,
          y=y_train,
          epochs=10,
          validation_data=(x_test, y_test),
          batch_size=10,
          callbacks=[checkpoint_5000])

# Saving weights with best validation accuracy
model = get_new_model()

model_path = generate_checkpoint_path('model_best')
checkpoint_best_acc = ModelCheckpoint(filepath=model_path,
                                      save_weights_only=True,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)
# test and save if acc improves
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=50,
                    validation_data=(x_test, y_test),
                    batch_size=10,
                    callbacks=[checkpoint_best_acc],
                    verbose=1)

df = pd.DataFrame(history.history)
df.plot(y=['accuracy', 'val_accuracy'])

model = get_new_model()
model.load_weights(filepath=model_path)
get_test_accuracy(model, x_test, y_test)  # should result in the 'best accuracy' shown while testing
