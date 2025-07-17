#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import math

# read a file, split words into list, convert them into lowercase 
def load_process_text(filename):
    """
    This function loads and reads given file

    Parameters
    ----------
    filename : file containing data for a model

    Returns
    -------
    processed_sentences : A list of sentences

    """
    try:
        with open(filename, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)
            processed_sentences = []
            
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                words = [word.lower() for word in tokens if word.isalpha()]
                if words:
                    processed_sentences.append(words)
            
    except OSError as e:
        print(f"Error reading a file: {e}")
        exit()
        
    return processed_sentences


def split_data(words, train_ratio=0.8):
    """
    This function splits dataset into training and testing 
    subsets for model evaluation

    Parameters
    ----------
    words : sentences for training 
     train_ratio : 
         float: ratio splits dataset into 80 % training subset.
         The default is 0.8.

     Returns
     -------
     train_sentences : list of list of sentences
     test_sentences : list of list of senteneces


    """
    split_point = int(train_ratio * len(words))
    train_sentences = words[:split_point]
    test_sentences = words[split_point:]
    return train_sentences, test_sentences

def bigram_probabilities_smoothed(train_sentences, alpha=0.1):
    """
    This function calculates bigram probabilities with smoothing

    Parameters
    ----------
    train_sentences : list of str containing training data
        
    alpha : Smoothing parameter The default is 0.1.

    Returns
    -------
    bigram_model : Nested dictionary of bigram probabilities 
    with smoothing

    """
    
    vocab = set(['<s>', '</s>'])
    for sentence in train_sentences:
        vocab.update(sentence)
    vocab_size = len(vocab)
    
    # Count bigrams and unigrams
    bigram_counts = defaultdict(lambda: defaultdict(int))
    unigram_counts = defaultdict(int)
    
    for sentence in train_sentences:
        words = ['<s>'] + sentence + ['</s>']
        for w1, w2 in zip(words, words[1:]):
            bigram_counts[w1][w2] += 1
            unigram_counts[w1] += 1
    
   
    bigram_model = defaultdict(lambda: defaultdict(float))
    
    for w1 in vocab:
        total_count = unigram_counts[w1]
        if total_count == 0:
            continue  
        
        for w2 in vocab:
            count = bigram_counts[w1][w2]
            prob = (count + alpha) / (total_count + alpha * vocab_size)
            bigram_model[w1][w2] = prob
    
    return bigram_model


def compute_joint_probab(sentence, bigram_probs):
    """
    Computes a joint probabilites for test corpus

    Parameters
    ----------
    test_sentences : list of list od str
    bigram_probs : dict: Nested disctionary containing bigram probabilities
        

    Returns
    -------
    dict: dictionary of probabilties of joint distribution

    """
    words = ['<s>'] + sentence + ['</s>']
    sentprob = 1.0
    
    for word1, word2 in zip(words, words[1:]):
        if word1 in bigram_probs and word2 in bigram_probs[word1]:
            sentprob *= bigram_probs[word1][word2]
        else:
            return 0.0  # Unseen bigram
    
    return sentprob

def compute_bigram_perplexity(test_sentences, bigram_probs):
    """
    Computes perplexity of a test corpus using a smoothed bigram model

    Parameters
    ----------
    test_sentences : list of list of str
    bigram_probs : dict: Nested disctionary containing bigram probabilities
        

    Returns
    -------
    TYPE
       float: The score for the test data

    """
    total_log_prob = 0.0
    total_words = 0
    
    for sentence in test_sentences:
        words = ['<s>'] + sentence + ['</s>']
        for w1, w2 in zip(words, words[1:]):
            if w1 in bigram_probs and w2 in bigram_probs[w1]:
                prob = bigram_probs[w1][w2]
                if prob <= 0:
                    prob = 1e-12  
                total_log_prob += math.log(prob)
                total_words += 1
         
    
    if total_words == 0:
        return float('inf')
    
    return math.exp(-total_log_prob / total_words)

def main():
    sentences = load_process_text('doyle_Bohemia.txt')
    train_sentences, test_sentences = split_data(sentences)
    
    # Build smoothed bigram model
    bigram_model = bigram_probabilities_smoothed(train_sentences)
    
    with open('smooth_probs.txt', 'w') as file:
        for w1 in sorted(bigram_model.keys()):
            for w2 in sorted(bigram_model[w1].keys()):
                prob = bigram_model[w1][w2]
                file.write(f"P({w2}|{w1}) = {prob}\n")
    
    # Evaluate test sentences
    with open('smoothed_eval.txt', 'w') as f:
        for sent in test_sentences:
            prob = compute_joint_probab(sent, bigram_model)
            f.write(f"p({' '.join(sent)}) = {prob}\n")
    
    # Calculate perplexity
    perplexity = compute_bigram_perplexity(test_sentences, bigram_model)
    print(f"Smoothed Bigram Model Perplexity: {perplexity}")

if __name__ == "__main__":
    main()     