import argparse
import os
import pickle
from pprint import pprint
import json
import numpy as np
from tqdm import tqdm

from imgsearch.dataset import Dataset
from imgsearch.search_engine.make_query import make_query

challenge_dataset_name = 'challenge'
group_name = 'Team01'

if __name__ == '__main__':

    os.system('clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description='Evaluate the accuracy of the search engine')

    parser.add_argument('-m', '--model', type=str, help='the model to use for the image search')
    parser.add_argument('-d', '--dataset', type=str, help='the dataset to evaluate the image search with')


    args = parser.parse_args()

    if args.dataset == challenge_dataset_name:
        gallery_dataset = Dataset(f'{args.dataset}/gallery')
        query_dataset = Dataset(f'{args.dataset}/query')

    else:
        gallery_dataset = Dataset(f'{args.dataset}/validation/gallery')
        query_dataset = Dataset(f'{args.dataset}/validation/query')

    with open(f'{args.dataset}_feats/{args.model}_query_feats.pkl', 'rb') as f:
        query_features = pickle.load(f)

    if gallery_dataset.labeled:

        all_t1, all_t3, all_t5, all_t10 = list(), list(), list(), list()

        for ind in tqdm(range(len(query_features)), desc='Testing on all queries'):
            t1, t3, t5, t10 = make_query(ind, args.model, gallery_dataset, query_dataset, quiet=True, show=False)

            all_t1.append(t1)
            all_t3.append(t3)
            all_t5.append(t5)
            all_t10.append(t10)

        print('RESULTS:')
        print(f'top-1  accuracy: {round(np.mean(all_t1), 4)}')
        print(f'top-3  accuracy: {round(np.mean(all_t3), 4)}')
        print(f'top-5  accuracy: {round(np.mean(all_t5), 4)}')
        print(f'top-10 accuracy: {round(np.mean(all_t10), 4)}')

    else:

        ret = {'groupname': group_name}

        all_searches = dict()

        for ind in tqdm(range(len(query_features)), desc='Testing on all queries'):
            cur_search = make_query(ind, args.model, gallery_dataset, query_dataset, quiet=True, show=False)

            all_searches.update(cur_search)

        ret['images'] = all_searches

        pprint(ret)

        with open(f'{args.model}_results.json', 'w') as f:
            json.dump(ret, f, indent=4)

