# Early stopping and patience
# Using callbacks allows one to monitor the performance metrics
# of the model and stop the training, if a chosen performance metric
# doesn't improve for some amount of iterations.

from course_utils import *
from tensorflow.keras.callbacks import EarlyStopping


unregularized_model = get_model()
unregularized_model.compile(optimizer='adam', loss='mse')

# monitor: what metric to look at
# patience: how many iterations without improvement until training stops
# min_delta: threshold to count 'improvement'
# mode: ['auto', 'min', 'max'], 'improvement' == increase or decrease of the metric?
unregularized_history = unregularized_model.fit(train_data, train_targets,
                                                epochs=100, validation_split=0.15,
                                                batch_size=64, verbose=False,
                                                callbacks=[EarlyStopping(monitor='loss',
                                                                         patience=5,
                                                                         min_delta=0)])
unregularized_model.evaluate(test_data, test_targets,
                             verbose=2)


regularized_model = get_regularized_model(1e-8, 0.2)
regularized_model.compile(optimizer='adam', loss='mse')
regularized_history = regularized_model.fit(train_data, train_targets,
                                            epochs=100, validation_split=0.15,
                                            batch_size=64, verbose=False,
                                            callbacks=[EarlyStopping(patience=5)])
regularized_model.evaluate(test_data, test_targets,
                           verbose=2)

plot_compare_metrics(unregularized_history, regularized_history, metric_name='loss')