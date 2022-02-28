import numpy as np

from course_utils import *
import tensorflow as tf
import json

model = Sequential([
    Dense(32, input_shape = (32, 32, 3),
          activation='relu'),
    Dense(10, activation='softmax')
])

config_dict = model.get_config()
print(config_dict)

# Create model with same architecture.
# Has not been trained yet though, so weights are different
model_same_config = tf.keras.Sequential.from_config(config_dict)
print('Same config:',
      model.get_config() == model_same_config.get_config())
print('Same value for 1st weight matrix:',
      np.allclose(model.weights[0].numpy(),
                  model_same_config.weights[0].numpy()))


# Saving architecture into json format:
json_string = model.to_json()
print(json_string)
save_path = generate_checkpoint_path('config.json')
with open(save_path, 'w') as f:
    json.dump(json_string, f)
del json_string

with open(save_path, 'r') as f:
    json_string = json.load(f)

# And check again
model_same_config = tf.keras.models.model_from_json(json_string)
print('Same config:',
      model.get_config() == model_same_config.get_config())
print('Same value for first weight matrix:',
      np.allclose(model.weights[0].numpy(), model_same_config.weights[0].numpy()))