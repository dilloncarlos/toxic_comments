# toxic_comments
For Kaggle contest: Jigsaw Toxic Comment Classification

## What's here so far?

There is an exploratory analysis kernel in jupyter notebook format found in 'Summary.ipynb'. The summary focuses on
correlations between the types of comment labels, missing / weird data, and most common terms in toxic comments.

## Classification: What is there and what to do?

Currently there is a binary relevance NB-SVM classifier of toxic comments, a LSTM NN classifier, and an 
"ensemble" script that averages the estimated probabilities from both. There are some obvious areas that can
be improved. Currently, the SVM assumes the labels are independent. You can see in the summary this is silly,
but recovering the lost correlation due to the binary relevance assumption is not trivial. Additionally, the LSTM
is far from optimal, though it currently performs quite well. 

## Who am I? 

I am a Graduate student and Data Science Initiative affiliate at UC Davis. I have played with some kaggle datasets 
before and used one competition as a project for a statistics course, but this is my first real structured attempt 
to compete. Ultimately, I hope to learn a lot and not finish in last place :P.
