# Sentence sentiment analysis
Used dataset of labelled sentences to learn a model and use the same to analyze the sentiment of new sentences
Dataset: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015

It contains sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants

**Format**:
sentence \t score \n

**Details**
Score is either 1 (for positive) or 0 (for negative)	
The sentences come from three different websites/fields:
* imdb.com
* amazon.com
* yelp.com

For each website, there exist 500 positive and 500 negative sentences, which are selected randomly.

