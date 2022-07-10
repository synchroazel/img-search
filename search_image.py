import argparse
import os
from pprint import pprint
from imgsearch.dataset import Dataset
from imgsearch.search_engine.make_query import make_query

challenge_dataset_name = 'challenge'

if __name__ == '__main__':
    os.system('clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description='Given a query image returns the 10 most similar from gallery')

    parser.add_argument('-i', '--query_index', type=int, help='index of the query image to query')
    parser.add_argument('-m', '--model', type=str, help='the model to use for the image search')
    parser.add_argument('-d', '--dataset', type=str, help='name of the dataset to use for the image search')
    parser.add_argument('-s', '--show', action='store_true', default=False, help='to show queried and fetched images')

    args = parser.parse_args()

    if args.dataset == challenge_dataset_name:
        gallery_dataset = Dataset(f'{args.dataset}/gallery')
        query_dataset = Dataset(f'{args.dataset}/query')

    else:
        gallery_dataset = Dataset(f'{args.dataset}/validation/gallery')
        query_dataset = Dataset(f'{args.dataset}/validation/query')

    res = make_query(args.query_index, args.model, gallery_dataset, query_dataset, show=args.show)

    pprint(res)
