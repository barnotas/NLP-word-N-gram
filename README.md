# NLP-text-N-gram
# Statistical Language Modelling with Unigram, Bigram, and Smoothed Bigram Models
# Project Description
This project implements and evaluates three statistical language models for estiimating the probability of natural language text. The models were trained and tested on a cleaned English corpus, and their performance was evaluated using perplexity as a metric. 
The key components of the project include:
* Text Preprocessing: Tokenization, lowercasing, punctuation removal, and sentence boundary detection were applied to raw text to prepare training and test dataset.
* Unigram Model
  This model computes the probability of each word independently using Maximum Likelihood Estimation (MLE).
* Bigram Model
  This model captures word-pair dependencies by computing the conditional probability of each word given its preceding word.
* Smoothed Bigram Model (Addititve Smoothing):
  To address the zero probability issue in bigram model, smoothing was impplemented. This technique adjusts the estimated probabilties to account for unseen word pairs.
* Evaluation
  All models were evaluated by computing the joint probabilities of test sentences and their coresponding perplexity scores.
