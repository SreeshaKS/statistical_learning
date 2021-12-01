
## Part 1: Part of Speech Tagging

Goal of this problem is to mark each word in the sentence with its part of speech.
The given training data has 12 parts of speeches and they are followed by the word in the sentence.
There are three different Baye's net model that needs to be modeled using training data and accuracy for the
not seen data has to be predicted using these models.

Using Training data, we found the initial probabilities, emission probability P(wi | si) and transition 
probabilities p(Si | Si-1). We also stored the part of speech tags and words in an dictianary along with
part of speech probabilities. Prbabilities for words which are not obderved in the training set are set to be
1/10^10.

Simplified model:
This model uses basics of Bayes Nets. Each word in the sentence is associated to only one part of speech tag. 
and select the best among them. THe max probability of the word given the part of speech is multiplied by 
part of speech probability to get the final results.

For the Viterbi model:
In this model, words have more than one independecy to each other than the simple word to pos dependecy. 
We used the maximum probability till each word in the sentence. and also the max proability in the state before
which are pre-calculated.

Posterior:
Simple: Calculated logs of emission probabilites and returned the sum
HMM Viterbi: Calculated logs of emission probabilities and transition probabilities
Complex: 

Assumptions:
For all the models, if we encounter a new word, emission probability was set to be 1/10^10
