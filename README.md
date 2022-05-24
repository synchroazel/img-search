
# Image search engine

An image search engine trained on 10 random animal classes from ImageNet, developed for a university project.

The following project had been proposed as a competition, in which each group had to build and train their engine and
test it on a new dataset on the competition deadline.

## Objective

The main objective of the project is to create an image search engine where a query image is fed to a model that will
return the most N similar images from a gallery. Given the input query image, the algorithm has to be capable of
matching the input query image with another gallery image depicting the same animal. The expected algorithm’s output is
a list of ranked matches between the query image and the gallery images.

## Project structure

```
./img-search/
│
├── .gitignore
├── README.md
├── img-search.ipynb     <- main notebook
├── notebook.html        <- rendered .html version of the notebook, for easy reading
├── challenge_01.json    <- challenge results with model1
├── challenge_02.json    <- challenge results with model2
└── assignment.pdf       <- the original assignment paper in .pdf
```

## Links

Download the (augmented) dataset here:<br>
https://drive.google.com/file/d/1fsltFvsFRNXP0TE_eG-_JXoGYy5PApQH/view?usp=sharing

Download the competition dataset here:<br>
https://drive.google.com/file/d/1mz1OgcrzbNsFC3stQWwadbUypAmeUopf/view?usp=sharing
