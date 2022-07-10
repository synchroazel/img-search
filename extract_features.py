import argparse
import pickle
import os

import tensorflow as tf
from tqdm import tqdm

from imgsearch.search_engine.features_extractor import featuresExtractor
from imgsearch.dataset import Dataset

from os import path

if __name__ == '__main__':

    os.system('clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    parser = argparse.ArgumentParser(description='Extract features from an image dataset')

    parser.add_argument('-d', '--dataset', type=str, help='path to the dataset to extract features from')
    parser.add_argument('-m', '--model', type=str, help='path to the model to use for features extraction')

    args = parser.parse_args()

    model = tf.keras.models.load_model(f'{args.model}')

    featExtractor = featuresExtractor(model)

    features = list()

    img_paths = Dataset(args.dataset).get_data_paths()

    for i in tqdm(range(len(img_paths)), desc=f'[INFO] Extracting features from {args.dataset}'):
        features.append(featExtractor.extract(img_paths[i]))

    model_name = path.basename(path.normpath(args.model))

    dataset_name = path.normpath(args.dataset).split("/")[0]
    subdataset_name = path.normpath(args.dataset).split("/")[-1]

    features_dir = f'{dataset_name}_feats'

    if features_dir not in os.listdir():
        os.mkdir(features_dir)

    pickle_name = path.join(features_dir, f'{model_name}_{subdataset_name}_feats') + '.pkl'

    with open(pickle_name, 'wb') as f:
        pickle.dump(features, f)