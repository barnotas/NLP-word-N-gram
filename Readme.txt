unigram perplexity score: 2760.41
bigram perplexity score: infinity
smoothed perplexity score: 388.90

---Which model performed worst and why might you have expected that model to have performed worst?
The bigram model perfomed worst with perplexity score of infinity
This means that any zero probability in the bigram model leads to a log probability of negative infinity, 
causing the overall perplexity to become infinite. In this case, the joint probability of the test sentences
under the bigram model was zero, which resulted in the highest possible perplexity score.

---Did smoothing help or hurt the model’s ‘performance’when evaluated on this corpus? Why might that be?
Smoothing technique definitely improved the model's performance. 
This technique helps the model by eliminating zero probabilities for word combinations not seen 
during training. By slightly adjusting the probabilities of observed bigrams, 
smoothing ensures that even unseen pairs are assigned a small, non-zero probability. 




