import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image


class featuresExtractor():

    def __init__(self, model):
        self.model = model

    def extract(self, img_path):
        '''
        Preprocess an image and fed it into the model to extract its features.
        '''

        # prepare image before model.predict()
        input_shape = (224, 224, 3)
        img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        # now we can .predict()
        features = self.model.predict(preprocessed_img, verbose=0)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)
        return normalized_features
