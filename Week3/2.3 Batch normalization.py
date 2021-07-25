# Batch Normalization
# It is conceptually a re-normalization of activations between layer outputs.
# Improves training speed and stability of the neural network.

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from course_utils import *

model = Sequential([
    Dense(64, input_shape=[train_data.shape[1], ], activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
])
model.summary()

# with specific params
model.add(tf.keras.layers.BatchNormalization(
    momentum=0.95,  # reusing previous running mean with some weight
    epsilon=0.005,  # controls numerical stability
    axis=-1,
    # beta and gamma perform additional affine transformation
    # by default beta=0, gamma=1
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
    gamma_initializer=tf.keras.initializers.Constant(value=0.9)
))
model.add(Dense(1))  # output layer

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

history = model.fit(train_data, train_targets,
                    epochs=100, validation_split=0.15,
                    batch_size=64, verbose=False)

plot_performance(history, metric_name='mae')
