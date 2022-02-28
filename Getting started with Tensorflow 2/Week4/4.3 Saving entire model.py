# We can also save the complete model,
# with weights and all... See below:

from course_utils import (get_new_model, generate_checkpoint_path, get_test_accuracy,
                          x_train, x_test, y_train, y_test)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

checkpoint_path = generate_checkpoint_path('model_checkpoints')
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_weights_only=False,
                             frequency='epoch',
                             verbose=1)
model = get_new_model()
model.fit(x=x_train,
          y=y_train,
          epochs=3,
          callbacks=[checkpoint])

get_test_accuracy(model, x_test, y_test)

del model
model = load_model(filepath=checkpoint_path)
get_test_accuracy(model, x_test, y_test)  # same accuracy

# Save in .h5 native format

new_path = generate_checkpoint_path('my_model.h5')
model.save(filepath=new_path)

del model
model = load_model(filepath=new_path)
get_test_accuracy(model, x_test, y_test)  # still the same
