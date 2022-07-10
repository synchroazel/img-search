import argparse
import os

from tqdm import tqdm

from imgsearch.dataset import Dataset
from imgsearch.preprocessing.augment import augment_img

if __name__ == '__main__':

    os.system('clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description='Augment a given dataset.')

    parser.add_argument('-d', '--dataset_path', type=str, help='the path of the dataset to augment')
    parser.add_argument('-n', '--augmentations', type=int, help='the number of augmentations to perform per image')

    args = parser.parse_args()

    to_augment = Dataset(args.dataset_path).get_data_paths()

    for img_path in tqdm(to_augment, desc=f'[TQDM] Augmenting images from {args.dataset_path}'):
        augment_img(img_path, args.augmentations)
