import os
import pickle
import random
import numpy as np
from numpy.linalg import norm
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from matplotlib import rcParams
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from imgsearch.dataset import Dataset
from imgsearch.search_engine.features_extractor import featuresExtractor


def make_query(query_index, model_name, gallery_dataset, query_dataset, quiet=False, show=False):

    dataset_name = gallery_dataset.data_path.split("/")[0]

    with open(f'{dataset_name}_feats/{model_name}_gallery_feats.pkl', 'rb') as f:
        gallery_features = pickle.load(f)

    with open(f'{dataset_name}_feats/{model_name}_query_feats.pkl', 'rb') as f:
        query_features = pickle.load(f)


    gallery_paths = gallery_dataset.get_data_paths()
    query_paths = query_dataset.get_data_paths()

    kNN_model = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')

    neighbors = kNN_model.fit(gallery_features)

    dists, ids = neighbors.kneighbors([query_features[query_index]])

    try:
        query_lab = query_dataset.data_mapping[query_paths[query_index]]

    except AttributeError:  # when gallery images are unlabeled
        pass

    if show:

        rcParams['figure.figsize'] = 7.5, 4.5

        f, axarr = plt.subplots(3, 5)

        axarr[0,2].imshow(mpimg.imread(query_paths[query_index]))
        axarr[0,2].set_title('Queried image')

        for c in range(5):
            axarr[0,c].axis('off')




        n = 0
        for r in range(1,3):
            for c in range(5):
                axarr[r, c].imshow(mpimg.imread(gallery_paths[ids[0][n]]))
                axarr[r, c].axis('off')
                n += 1

        plt.show()



        # plt.imshow(mpimg.imread(query_paths[query_index]))
        # plt.title('Queried image', {'color': 'white'})  # for dark mode!
        # plt.axis('off')
        #
        # plt.show()
        #
        # rcParams['figure.figsize'] = 15, 5
        #
        # f, axarr = plt.subplots(2, 5)
        #
        # n = 0
        # for r in range(2):
        #     for c in range(5):
        #         axarr[r, c].imshow(mpimg.imread(gallery_paths[ids[0][n]]))
        #         axarr[r, c].axis('off')
        #         n += 1
        #
        # plt.show()

    cur_key = gallery_paths[query_index].split('/')[-1]

    if query_dataset.labeled:


        if not quiet:

            print(f'\nQueried img:\n{query_lab}')
            print('\nRetrieved imgs:')

            for ind in ids[0]:

                gallery_lab = gallery_dataset.data_mapping[gallery_paths[ind]]

                if query_lab == gallery_lab:
                    print(gallery_lab, '○')
                else:
                    print(gallery_lab, '✕')

        matches = list()


        for ind in ids[0]:

            gallery_lab = gallery_dataset.data_mapping[gallery_paths[ind]]

            if query_lab == gallery_lab:
                matches.append(True)
            else:
                matches.append(False)

        in_top1, in_top3, in_top5, in_top10 = any(matches[:1]), any(matches[:3]), any(matches[:5]), any(matches[:10])

        return in_top1, in_top3, in_top5, in_top10

    else:

        ret = {cur_key: []}

        for ind in ids[0]:
            ret[cur_key].append(gallery_paths[ind].split('/')[-1])

        return ret
