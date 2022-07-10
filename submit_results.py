import argparse
import json

import requests


def submit(results, url='https://tinyurl.com/IML2022'):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit results .json file to the evaluation server for the challenge')

    parser.add_argument('-r', '--results', type=str, help='name of the .json file of results to submit')

    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    print(submit(results))
