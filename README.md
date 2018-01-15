# Neural-Review-Summarization: Summarization of the essence of user movie reviews

## Overview

A neural model that extracts the essence out of movie reviews. The model
first looks through a large number of movie reviews, forms different
n-grams (e.g. unigrams / bigrams / trigrams) and uses POS tagging to
filter out the meaningful n-grams for movie reviews. Once those n-grams
are filtered, GloVe embeddings and AffinityPropagation Clustering is used
for grouping the remaining n-grams into meaningful clusters. Meaningful
clusters are later on labeled by hand (600-1000 clusters) for their
relevance to a domain. During prediction time the clusters are used
with K-Nearest-Neighbours algorithm to predict which of the new phrases
are meaningful. Additional filtering, compression and recommendations
are applied before presenting the final result.

This model is inspired by the GooglePlay model for review summarization:

![picture](png/google-play.png)

### Example

Input: [80 Midnight Cowboy User Reviews](http://www.imdb.com/title/tt0064665/reviews?ref_=tt_ov_rt)

Output:
```
[{'count': 2, 'n_size': 3, 'phrase': 'high quality film', 'score': 2.8},
 {'count': 8, 'n_size': 2, 'phrase': 'beautiful film', 'score': 11.2},
 {'count': 7, 'n_size': 2, 'phrase': 'fine film', 'score': 9.8},
 {'count': 6, 'n_size': 2, 'phrase': 'academy award', 'score': 6.48},
 {'count': 4, 'n_size': 2, 'phrase': 'sad film', 'score': 5.6},
 {'count': 4, 'n_size': 2, 'phrase': 'fabulous movie', 'score': 5.6},
 {'count': 4, 'n_size': 2, 'phrase': 'sexual abuse', 'score': 5.6},
 {'count': 3, 'n_size': 2, 'phrase': 'sexual feelings', 'score': 4.2},
 {'count': 3, 'n_size': 2, 'phrase': 'powerful film', 'score': 4.2},
 {'count': 3, 'n_size': 2, 'phrase': 'insightful character', 'score': 4.2},
 {'count': 3, 'n_size': 2, 'phrase': 'only way', 'score': 3.24},
 {'count': 2, 'n_size': 2, 'phrase': 'outstanding performances', 'score': 2.8},
 {'count': 2, 'n_size': 2, 'phrase': 'sexual revolution', 'score': 2.8},
 {'count': 2, 'n_size': 2, 'phrase': 'sexual theatrics', 'score': 2.8},
 {'count': 2, 'n_size': 2, 'phrase': 'stellar performances', 'score': 2.8},
 {'count': 2, 'n_size': 2, 'phrase': 'sympathetic handsome', 'score': 2.8},
 {'count': 54, 'n_size': 1, 'phrase': 'good', 'score': 58.33},
 {'count': 53, 'n_size': 1, 'phrase': 'great', 'score': 57.24},
 {'count': 43, 'n_size': 1, 'phrase': 'big', 'score': 53.32},
 {'count': 23, 'n_size': 1, 'phrase': 'bad', 'score': 24.85}]
```

## Version

Current version : __***0.0.0.1***__

## Dependencies ( VERSION MUST BE MATCHED EXACTLY! )
To Be Added

## Installation
To Be Added

## Jupyter Notebooks
to use jupyter notebooks in the created virtual environment, follow
these instructions: [https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/](instructions)

## Training the model

The model is trained in mostly unsupervised manor with a small dataset
that needs to be labeled by hand (600-1000 entries). During training
n-grams (phrases) are extracted from the 25,000 IMDB reviews and then
they are preprocessed and clustered. The cluster centers and labels of
each phrase are the output of the model - they are saved in a csv file
for later labeling and usage.

For more information about the labeling strategy check `model/saved`
directory

![picture](png/model-training.png)

## Predicting

Predictions go through several steps:
1. Pre-processing - makes sure that movies' reviews are broken down into
n-grams and all clutter is removed.
1. Affinity Model Predictions - predicts the cluster ids of all the
remaining n-grams. N-grams that do not fall into the good clusters group
are filtered out.
1. Summary Recommendations - applies additional steps to filter, compress
and recommend the best summaries

![picture](png/model-predictions.png)

### Pre-processing

The `IMDBPreprocessor` has the following public APIs:
1. `load_data` - loads a csv / tsv file with reviews related to a movie
1. `prepare_data` - splits all the data into n-grams, POS tagging and
POS template filtering

Additionally `unigrams`, `bigrams` and `trigrams` are also public and
ready to use after `prepare_data` has been called.

### Affinity Model Predictions

The `AffinityClusterModel` has the following important public APIs:
1. `save_model` and `load_model`
1. `predict` - given n-gram phrases - vectorizes them and uses KNN
to predict the most appropriate cluster label
1. `get_phrases_in_good_clusters` - filters any predictions that do
not fall into the manually labeled good clusters

![picture](png/affinity-model.png)

### Summary Recommendations

The `SummaryRecommender` model has the following important APIs:
1. `generate_phrases` - loads reviews about a movie from a file,
internally calls the `IMDBPreprocessor` APIs
1. `recommend_phrases` - suggests recommended summaries. Internally
calls `AffinityClusterModel` and then applies Blacklist Filtering,
Compression and Summary Recommendations on top

![picture](png/recommender-model.png)

## Future works
To Be Added

## Authors
George Stoyanov (georgi.val.stoyan0v@gmail.com)