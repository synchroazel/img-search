
# Image Search Engine

An image search engine trained on 10 random animal classes from ImageNet, developed for a university project.

The following project had been proposed as a competition, for a Machine Learning introductory course at University of Trento (MSc in Data Science). 

## Objective

The main objective of the project is to create an image search engine where a query image is fed to a model that will
return the most N similar images from a gallery. Given the input query image, the algorithm has to be capable of
matching the input query image with another gallery image depicting the same animal. The expected algorithm’s output is
a list of ranked matches between the query image and the gallery images.

## Usage (quick guide)

A quick recap of the main usage of the different modules. The next section explains more in depth the rationale behind each component functioning and the arguments required.

```bash
# AUGMENTATION
python3 augment_dataset.py -d <path to dataset> -n <n. of augmentations>
	
# TRAINING
python3 train_<model>.py -d <path to training dataset>
	
# FEATURES EXTRACTION
python3 extract_features.py -d <path to dataset> -m <path to model>

# SEARCH
python3 search_image.py -m <model> -d <dataset> -i <query img index>
	
# EVALUATION
python3 evaluate_engine.py -d <dataset> -m <model>
```

## Usage (explained)

### 1. Data augmentation

The initial dataset can be augmented using `augment_dataset.py`

```bash
python3 augment_dataset.py \
	-d <path to dataset \
	-n <n. of augmentations>
```

where `-d` specify the path to the dataset to augment, and `-n` is the number of augmentations to apply for each image. 

To augment the training dataset with 5 random augmentations for each image, use:

```bash
python3 augment_dataset.py \
	-d dataset/training \
	-n 5
```

### 2. Training

Two models were considered for the following project:

- a model based on **ResNet152**
- a built-from-scratch Convolutional Neural Network

Each model can be instantiated and trained through `train_<model>.py`, namely for the Resnet-backbone model:

```bash
python3 train_resnet.py \
	-d dataset/training
```

and for the custom model:

```bash
python3 train_custom.py \
	-d dataset/training
```

The `-d` argument refers to the path of the training dataset to use.

In both cases, trained models are saved to `models/` for later re-usage.

You can also manually specify with `-e` the number of epochs for the training phase.

```bash
python3 train_custom.py \
	-d dataset/training
	-e 100
```

### 3. Features extraction

To extract features from a given dataset using a given model, use:

```bash
python3 extract_features.py \
	-d <path to dataset> \
	-m <path to model>
```

So, for example, to extract features from gallery and query images from our actual dataset, use:

```bash
# USING resnet

python3 extract_features.py \
	-d dataset/validation/query \
	-m models/resnet

python3 extract_features.py \
	-d dataset/validation/gallery \
	-m models/resnet

# USING myconv

python3 extract_features.py \
	-d dataset/validation/query \
	-m models/myconv

python3 extract_features.py \
	-d dataset/validation/gallery \
	-m models/myconv
```

Features will be saved in `features/` to be accessed by the actual search engine.

### 4. Search engine

You can actually perform a search with the following:

```bash
python3 search_image.py \
    -i <query img index> \
	-m <model> \
	-d <dataset>
```

where `-i` is the index of the query image to search and `-m` is the chosen model.

An example search (for query image #10 using Resnet model) would be:

```bash
python3 search_image.py \
    -i 10 \
	-m resnet
	-d dataset
```

Notice that the evaluation proceeds in 2 different ways according to the images being labeled or not.

#### labeled images

If images are labeled, the script simply search for the 10 gallery images most similar to the query one, comparing the features extracted before, and returns the top1-3-5-10 accuracies for the current search (using labels to identify a hit/miss).

#### unlabeled images

If images are unlabeled (meaning the models are being tested on new challenge data) the script simply returns a dictionary mapping the query image to an ordered list of the 10 most similar gallery images.

### 5. Evaluation

To evaluate a model on all query images, simply use:

```bash
python3 evaluate_engine.py \
	-m <model>
	-d <dataset>
```

So, for example, to evaluate the Resnet model use:

```bash
python3 evaluate_engine.py \
	-m resnet
	-d dataset
```

Notice that the evaluation proceeds in 2 different ways according to the images being labeled or not.

#### labeled images

If images are labeled, the script simply repeats the search process for each query image, storing the accuracies obtained and returns an average of top1-3-5-10 accuracies.

#### unlabeled images

If images are unlabeled (meaning the models are being tested on new challenge data) the script stores in a `results.json` the mapping of each query image to the corresponding top10 most similar images.

Such file can be submitted to the University server used for the evaluation on the challenge deadline day.

### 6. Submit results

Finally, to submit the results of a model on unlabeled competition data to the university server used for the evaluation, use `submit_results.py` as follows:

```bash
python3 submit_results.py \
	-r <path to results>
```

where `-r` is used to specify the path to the .json file with results.

## Links

Download the original dataset here:<br>
https://drive.google.com/file/d/1T-d3gHIIaovaViE0e0o8XW8Rlh49OMC7/view?usp=sharing

Download the challenge dataset here:<br>
https://drive.google.com/file/d/1mz1OgcrzbNsFC3stQWwadbUypAmeUopf/view?usp=sharing
