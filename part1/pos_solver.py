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
        self.P_DEFAULT = 0.0000000000000000000000000001
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

    def posterior(self, model, sentence, label):

        if model == "Simple":
            posterior = 0
            for i in range(len(sentence)):
                if (sentence[i], label[i]) in self.p_emission:
                    posterior += math.log(self.p_emission[(sentence[i], label[i])])
                else:
                    posterior += self.minProb  # setting min prob value for words which are not in training data
                posterior += math.log(self.pos[label[i]])
            return posterior

        elif model == "HMM":
            posterior = 0
            for i in range(len(sentence)):
                if (sentence[i], label[i]) in self.p_emission:
                    posterior += math.log(self.p_emission[(sentence[i], label[i])])

                else:
                    posterior += math.log(self.p_transition[(label[i], label[i - 1])])
            return posterior

        elif model == "Complex":
            return sum(self.generateSampleProbabilites(sentence, label))

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
        # Creating VITERBI TABLE for Viterbi Algorithm

        viterbi = [[0] * len(self.pos) for i in range(len(sentence))]
        unique_tags = []
        for key in self.pos:
            unique_tags.append(key)
        pos_word_mapping = []
        pos_word_mapping_01 = []
        for i in range(len(sentence)):
            for j in range(len(unique_tags)):
                if (sentence[i], unique_tags[j]) in self.p_emission:
                    pb = self.p_emission[(sentence[i], unique_tags[j])]
                else:
                    pb = self.minProb

                if i == 0:  # for initial word
                    viterbi[i][j] = self.p_initial[unique_tags[j]] * pb

                else:  # for all subsequent words
                    x = 0
                    y = 0

                    for v in viterbi[i - 1]:
                        v_trans = v * self.p_transition[(unique_tags[x], unique_tags[j])]
                        if y < v_trans:
                            y = v_trans
                        x += 1
                    viterbi[i][j] = pb * y
        # pos prabables 
        for i in range(len(viterbi) - 1, -1, -1):
            pos_word_mapping.append(unique_tags[viterbi[i].index(max(viterbi[i]))])
        for i in range(len(pos_word_mapping) - 1, -1, -1):
            pos_word_mapping_01.append(pos_word_mapping[i])
        return pos_word_mapping_01

    def generateSampleProbabilites(self, sentence, pos_tags):
        P_DEFAULT = self.P_DEFAULT
        p_samples = []
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
            elif index < len(sentence) - 1:
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
        return p_samples

    # Discussed with Zack and Sylvia
    # Ideas from Canvas lectures
    # Findings from - Bayesian Networks: MCMC with Gibbs Sampling -  https://www.youtube.com/watch?v=dZoHsVO4F3k
    # Partially implemented
    def complex_mcmc(self, sentence):
        MAX_ITERATIONS = 5000

        pos_tags = self.simplified(sentence)
        final = {}
        for step in range(MAX_ITERATIONS):
            p_samples = self.generateSampleProbabilites(sentence, pos_tags)
            Sum = sum(p_samples)
            p_samples = [sample / Sum for sample in p_samples]
            flip = random.uniform(0, 1)
            p_max = 0
            for _, sample in enumerate(p_samples):
                p_max *= sample
                if p_max > flip:
                    # pick a part of speech here - to be implemented
                    pos_tags[index] = 'noun'
                    break
            for index, state_pos in enumerate(pos_tags):
                if (index, pos_tags[index]) not in final:
                    final[(index, pos_tags[index])] = 1
                else:
                    final[(index, pos_tags[index])] += 1
        labels = []
        for index, state_pos in enumerate(pos_tags):
            p_max = 0
            state_pos = 'noun'
            for pos in self.pos:
                if (index, pos) in final:
                    if p_max < final[(index, pos)]:
                        p_max = final[(index, pos)]
                        state_pos = pos
            labels.append(state_pos)

        return labels

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
