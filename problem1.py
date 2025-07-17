#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from nltk.tokenize import sent_tokenize, word_tokenize
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


def calculate_unigram_probs(train_words):
    """
    This function calculates unigram probabilities 

    Parameters
    ----------
    train_sentences : list of words containing training data
        

    Returns
    -------
    unigram_probs : dictionary of unigram probabilities 
    

    """
    word_count = {word: 0 for word in train_words}
    for w in train_words:
        if w in word_count:
            word_count[w] += 1
        else:
            word_count[w] = 1
            
    count = sum(word_count.values())

    unigram_probs = {word: word_count[word]/count for word in word_count}
    return unigram_probs 


def compute_joint_probab(sentences, unigram_probs):
    """
    Computes a joint probabilites for test corpus

    Parameters
    ----------
    test_sentences : list of list of sentences
    unigram_probs : dict: disctionary containing unigram probabilities
        

    Returns
    -------
    dict: dictionary of probabilties of joint distribution

    """
    sentprob = 1
    for word in sentences:
        if word in unigram_probs:
            sentprob *= unigram_probs[word]
        else:
            return 0
    return sentprob


def compute_perplexity(test_data, unigram_probs):
    """
    Computes perplexity of a test corpus using a unigram model

    Parameters
    ----------
    test_sentences : list of list of sentences
    unigram_probs : dict:  disctionary containing bigram probabilities
        

    Returns
    -------
    TYPE
       float: The score for the test data

    """
    log_prob_sum = 0
    word_count = 0
    
    for sentence in test_data:
        for word in sentence:
            if word in unigram_probs:
                log_prob_sum += math.log(unigram_probs[word])
            else:
                # If word is unseen, assign a small probability (Laplace smoothing alternative)
                log_prob_sum += math.log(1e-10)  # avoid log(0)
            word_count += 1
            
    return math.exp(-log_prob_sum / word_count) if word_count > 0 else float('inf')

    

    

def main():
    sentences  = load_process_text('doyle_Bohemia.txt')
    train_dat, test_dat = split_data(sentences)
    unigram_probs = calculate_unigram_probs(train_dat)
    with open('unigram_probs.txt', 'w') as file:
        for word, prob in unigram_probs.items():
            file.write(f"p({word}) = {prob}\n")
    
    with open('unigram_eval.txt', 'w') as f:
        for sent in test_dat:
            prob = compute_joint_probab(sent, unigram_probs)
            f.write(f"p({' '.join(sent)}) = {prob}\n")
    perplexity = compute_perplexity(test_dat, unigram_probs)
    print(f"Unigram Model Perplexity: {perplexity}")  
if __name__ == "__main__":
    main()     