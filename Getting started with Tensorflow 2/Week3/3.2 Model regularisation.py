# Regularization
# Regularization is a collective of methods which allow one to
# stabilize the training, and reduce the risk of over-fitting.

from course_utils import *


def get_regularized_model(wd, rate):
    model_ = Sequential([
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
    return model_


model = get_regularized_model(1e-5, 0.1)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(train_data, train_targets, epochs=100,
                    validation_split=0.15, batch_size=64, verbose=False)
model.evaluate(test_data, test_targets, verbose=2)
plot_performance(history, metric_name=None)
