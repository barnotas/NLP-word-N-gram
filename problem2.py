#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import math


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



def split_data(words, train_ratio= 0.8):
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
    split_data = int(train_ratio * len(words))
    train_words = words[:split_data]
    train_words_flat = [word for sentence in train_words for word in sentence]
    test_words = words[split_data:]
    return train_words_flat, test_words



def bigram_probablities(train_dat):
    
    """
    This function calculates bigram probabilities

    Parameters
    ----------
    train_sentences : list of words containing training data
        
    alpha : Smoothing parameter The default is 0.1.

    Returns
    -------
    bigram_model : dictionary of bigram probabilities 

    """
    bigram_model = defaultdict(lambda: defaultdict(int))
    words = ['<s>'] + train_dat + ['</s>']
    for w1, w2 in zip(words, words[1:]):
        bigram_model[w1][w2] += 1


    
    # Compute MLE probabilties 
    for w1 in bigram_model:
        total_count = float(sum(bigram_model[w1].values()))
        for w2 in bigram_model[w1]:
            bigram_model[w1][w2] /= total_count
    return bigram_model

def compute_joint_probab(sentences, bigram_probs):
    """
    Computes a joint probabilites for test corpus

    Parameters
    ----------
    test_sentences : list of list od str
    bigram_probs : dict: disctionary containing bigram probabilities
        

    Returns
    -------
    dict: dictionary of probabilties of joint distribution

    """
    words = ['<s>'] + sentences + ['</s>']
    sentprob = 1
    for word1, word2 in zip(words, words[1:]):
        if word1 in bigram_probs and word2 in bigram_probs[word1]:
            sentprob *= bigram_probs[word1][word2]
        else:
            return 0.0
    return sentprob

def compute_bigram_perplexity(test_data, bigram_probs):
    """
    Computes perplexity of a test corpus using a bigram model

    Parameters
    ----------
    test_sentences : list of list of str
    bigram_probs : dict: disctionary containing bigram probabilities
        

    Returns
    -------
    TYPE
       float: The score for the test data

    """
    total_log_prob = 0.0
    total_words = 0
    
    for sentence in test_data:
        words = ['<s>'] + sentence + ['</s>']
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 in bigram_probs and w2 in bigram_probs[w1]:
                prob = bigram_probs[w1][w2]
                total_log_prob += math.log(prob)
                total_words += 1
            else:
                return float('inf')
    
    if total_words == 0:
        return float('inf')
    
    return math.exp(-total_log_prob / total_words)




def main():
    words = load_process_text('doyle_Bohemia.txt')
    train_dat, test_dat = split_data(words)
    bigram_model = bigram_probablities(train_dat)
    with open('bigram_probs.txt', 'w') as file:
        for w1 in bigram_model:
            for w2 in bigram_model[w1]:
                prob = bigram_model[w1][w2]
                file.write(f"P({w2}|{w1}) = {prob}\n")
    with open('bigram_eval.txt', 'w') as f:
        for sent in test_dat:
            prob = compute_joint_probab(sent, bigram_model)
            f.write(f"p({' '.join(sent)}) = {prob}\n")
    perplexity = compute_bigram_perplexity(test_dat, bigram_model)
    print(f"Bigram Model Perplexity: {perplexity}")

if __name__ == "__main__":
    main()     