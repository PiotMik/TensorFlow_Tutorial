from course_utils import *

model = get_model()
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(train_data, train_targets,
                    epochs=100, validation_split=0.15,
                    batch_size=64, verbose=False)

model.evaluate(test_data, test_targets, verbose=2)
plot_performance(history)
