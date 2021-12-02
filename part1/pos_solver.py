# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:56:03 2021
@author: Amruta
"""

import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def __init__(self):
        self.minProb = float(1) / float(10 ** 10)
        self.p_initial = {}
        # transition probabilities
        self.p_transition = {}

        # P(wi|posTag) - emission probability
        self.p_emission = {}
        # total number of words which occur under a particular tag
        self.words = {}
        # total number of pos tags
        self.pos = {}
        #

    def posterior(self, model, sentence, label):
        # posterior = 0

        # for i in range(len(sentence)):
        #     if (sentence[i], label[i]) in self.p_emission:
        #         prob= self.p_emission[(sentence[i],label[i])]
        #     else:
        #         prob=self.minProb  #setting min prob value for words which are not in training data
        #     posterior+=math.log(prob*self.pos[label[i]])

        if model == "Simple":
            posterior = 0
            for i in range(len(sentence)):

                if (sentence[i], label[i]) in self.p_emission:
                    prob = self.p_emission[(sentence[i], label[i])]
                else:
                    prob = self.minProb  # setting min prob value for words which are not in training data
                posterior += math.log(prob * self.pos[label[i]])
            return posterior
        elif model == "HMM":
            posterior = 0
            for i in range(len(sentence)):
                if (sentence[i], label[i]) in self.p_emission:
                    posterior += math.log(self.p_emission[(sentence[i], label[i])])
                # if i == 0:
                # posterior += math.log(self.transition_prob[label[i]][" "])
                # posterior += math.log(self.p_transition[(label[i], "")])
                else:
                    # posterior += math.log(self.transition_prob[label[i]][label[i-1]])
                    posterior += math.log(self.p_transition[(label[i], label[i - 1])])

            return posterior

        elif model == "Complex":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):

        total_pos_count = 0
        for sentence in data:
            dict_words = sentence[0]

            dict_pos = sentence[1]
            for i in range(0, len(dict_words)):
                # Updating words count
                if dict_words[i] not in self.words:
                    self.words.update({dict_words[i]: 1})
                else:
                    self.words[dict_words[i]] = self.words[dict_words[i]] + 1
                # updating pos tags count
                if dict_pos[i] not in self.pos:
                    self.pos.update({dict_pos[i]: 1})
                    total_pos_count += 1
                else:
                    self.pos[dict_pos[i]] = self.pos[dict_pos[i]] + 1
                    total_pos_count += 1
                # Initial pos tags count
                if i == 0:
                    if dict_pos[i] not in self.p_initial:
                        self.p_initial.update({dict_pos[i]: 1})
                    else:
                        self.p_initial[dict_pos[i]] = self.p_initial[dict_pos[i]] + 1
                else:
                    # Other than initial are set as 0
                    if dict_pos[i] not in self.p_initial:
                        self.p_initial.update({dict_pos[i]: 0})
                        # Transition prob count
                    elif (dict_pos[i - 1], dict_pos[i]) not in self.p_transition:
                        self.p_transition.update({(dict_pos[i - 1], dict_pos[i]): 1})
                    else:
                        self.p_transition[(dict_pos[i - 1], dict_pos[i])] = self.p_transition[
                                                                                (dict_pos[i - 1], dict_pos[i])] + 1
                # emission probability count

                if (dict_words[i], dict_pos[i]) not in self.p_emission:
                    self.p_emission.update({(dict_words[i], dict_pos[i]): 1})
                else:
                    self.p_emission[(dict_words[i], dict_pos[i])] = self.p_emission[(dict_words[i], dict_pos[i])] + 1

        # initial probability
        for key, value in self.p_initial.items():
            if value == 0:
                self.p_initial[key] = self.minProb
            else:
                self.p_initial[key] = float(self.p_initial[key]) / float(self.pos[key])
        # emission probability
        for key in self.p_emission:
            self.p_emission[key] = float(self.p_emission[key]) / float(self.pos[key[1]])
            # Values that dont exist in training set
        for i in self.words:
            for j in self.pos:
                if (i, j) not in self.p_emission:
                    self.p_emission.update({(i, j): self.minProb})

                    # transition probability
        for key in self.p_transition:
            self.p_transition[key] = float(self.p_transition[key]) / float(self.pos[key[0]])

        # For values that dont exist in training set
        for i in self.pos:
            for j in self.pos:
                if (i, j) not in self.p_transition:
                    self.p_transition.update({(i, j): self.minProb})

        # pos probability
        for key in self.pos:
            self.pos[key] = float(self.pos[key]) / float(total_pos_count)
        print('Done Training')

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        pos_word_mapping = []
        for word in sentence:
            max_probability = 0
            pos_word = ""
            for pos in self.pos:
                if (word, pos) in self.p_emission:
                    pb = self.p_emission[(word, pos)] * self.pos[pos]
                else:
                    pb = self.minProb
                if pb > max_probability:
                    max_probability = pb
                    pos_word = pos
            pos_word_mapping.append(pos_word)
        return pos_word_mapping

    def hmm_viterbi(self, sentence):
        most_probable_pos = []
        most_probable_pos_return = []

        # Table to implement Viterbi Algorithm
        viterbi_table = [[0] * len(self.pos) for i in range(len(sentence))]
        pos_list = []

        # List of unique tags
        for key in self.pos:
            pos_list.append(key)

        # Viterbi Algorithm
        for i in range(0, len(sentence)):
            for j in range(0, len(pos_list)):
                if (sentence[i], pos_list[j]) not in self.p_emission:
                    prob = 0.0000000001
                else:
                    prob = self.p_emission[(sentence[i], pos_list[j])]

                # For first word, Vij=P(Si)*P(Wi|Si)
                if i == 0:
                    viterbi_table[i][j] = self.p_initial[pos_list[j]] * prob

                # For words that aren't the first, Vij=P(Wi|Si)*max(VIJ*P(SI|SI-1))
                else:
                    k = 0
                    max_vt = 0.0
                    for v in viterbi_table[i - 1]:
                        v_transition = v * self.p_transition[(pos_list[k], pos_list[j])]
                        if max_vt < v_transition:
                            max_vt = v_transition
                        k = k + 1
                    viterbi_table[i][j] = prob * max_vt

        # Extracting the most probable tag fromeh viterbi table
        for i in range(len(viterbi_table) - 1, -1, -1):
            most_probable_pos.append(pos_list[viterbi_table[i].index(max(viterbi_table[i]))])
        for i in range(len(most_probable_pos) - 1, -1, -1):
            most_probable_pos_return.append(most_probable_pos[i])

        return most_probable_pos_return
        # return ["noun"] * len(sentence)

    def complex_mcmc(self, sentence):
        MAX_ITERATIONS = 5000
        P_DEFAULT = 0.0000000000000000000000000001
        pos_tags = self.simplified(sentence)
        for step in range(MAX_ITERATIONS):
            for index, word in enumerate(sentence):
                p_samples = []
                if len(sentence) == 1:
                    for state_pos in self.pos:
                        p_samples += [
                            self.p_emission.get((word, state_pos), P_DEFAULT)
                            * self.p_initial[state_pos]
                        ]
                elif index == 0:
                    for state_pos in self.pos:
                        p_samples += [
                            self.p_emission.get((word, state_pos), P_DEFAULT)
                            * self.p_initial[state_pos]
                            * self.p_transition.get((state_pos, pos_tags[index + 1]), P_DEFAULT)
                        ]
                elif word == len(pos_tags) - 1:
                    for state_pos in self.pos:
                        p_samples += [
                            self.p_emission.get((word, state_pos), P_DEFAULT)
                            * self.p_transition.get((pos_tags[index - 1, pos_tags[0]]), P_DEFAULT)
                            * self.p_emission.get((word, pos_tags[index - 1]), P_DEFAULT)
                            * self.p_transition.get((state_pos, pos_tags[index - 1]), P_DEFAULT)
                        ]
                elif index < len(sentence)-1:
                    for state_pos in self.pos:
                        p_samples += [
                            self.p_emission.get((word, state_pos), P_DEFAULT)
                            * self.p_emission.get((sentence[index - 1], pos_tags[index - 1]), P_DEFAULT)
                            * self.p_emission.get((word, pos_tags[index - 1]), P_DEFAULT)
                            * self.p_emission.get((sentence[index + 1], pos_tags[index]), P_DEFAULT)
                            * self.p_emission.get((word, pos_tags[index + 1]), P_DEFAULT)
                            * self.p_transition.get((sentence[index - 1], state_pos), P_DEFAULT)
                            * self.p_transition.get((state_pos, pos_tags[index + 1]), P_DEFAULT)
                        ]
                Sum = sum(p_samples)
                p_samples = [sample / Sum for sample in p_samples]
                flip = random.uniform(0, 1)
                # create weighted sample
        return ["noun"] * len(sentence)

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")
