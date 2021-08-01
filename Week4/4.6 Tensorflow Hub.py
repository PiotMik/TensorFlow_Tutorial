# Tensorflow Hub
# It's a 'storage' of useful modules for specific tasks.
# You can download them and use for your use
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
module_url = "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/4"
model = Sequential([hub.KerasLayer(module_url)])
model.build(input_shape=[None, 160, 160, 3])

lemon_img = load_img("data/lemon.jpg", target_size=(160, 160))
viaduct_img = load_img("data/viaduct.jpg", target_size=(160, 160))
water_tower_img = load_img("data/water_tower.jpg", target_size=(160, 160))

with open('data/imagenet_categories.txt') as txt_file:
    categories = txt_file.read().splitlines()


def get_top_5_predictions(img):
    x = img_to_array(img)[np.newaxis, ...] / 255.0
    preds = model.predict(x)
    top_preds = pd.DataFrame(columns=['prediction'],
                             index=np.arange(5) + 1)
    sorted_index = np.argsort(-preds[0])
    for i in range(5):
        ith_pred = categories[sorted_index[i]]
        top_preds.loc[i + 1, 'prediction'] = ith_pred

    return top_preds

get_top_5_predictions(lemon_img)
get_top_5_predictions(viaduct_img)
get_top_5_predictions(water_tower_img)