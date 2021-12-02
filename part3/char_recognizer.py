from collections import defaultdict
import math
import sys

TRAIN_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' ")
CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25
max_val = sys.maxsize 
min_val = sys.float_info.epsilon

class CharacRecognizer:

    def __init__(self, train_letters, test_letters, train_txt_fname):
        
        self.train_letters = train_letters
        self.test_letters = test_letters
        self.train_txt_fname = train_txt_fname
        # Intitial probability dictionary stores the number of times a character appears in the training set on the first position of a new word
        self.init_prob = dict()
        
        # Emmision probability dictionary stores the emmision probability, i.e. probabalities of TRAIN_LETTERS being related to the Traing picture provided. In this case - courier.png
        # Initialized to about 0 prob errors
        self.emission_prob = defaultdict()
        for i in range(len(test_letters)):
            self.emission_prob[i] = dict()
            for j in TRAIN_LETTERS:
                self.emission_prob[i][j] = max_val

        # Transition probability dictionary stored the probabily of characters appearing next to each other in the training set. 
        # Initialized to about 0 prob errors
        self.transition_prob = defaultdict()
        for i in TRAIN_LETTERS:
            self.transition_prob[i] = dict()
            for j in TRAIN_LETTERS:
                self.transition_prob[i][j] = min_val
        
        # Train for calculating the probabilities
        self.train(train_txt_fname)

    # Compares two letters and returns the number of matching pixels and the number of blanks
    def compare_letters(self, letter1, letter2):
        matches = 0
        blanks = 0
        for row1, row2 in zip(letter1, letter2):
            for ch1, ch2 in zip(row1, row2):
                if ch1 == ch2 == '*':
                    matches += 1
                if ch1 == ch2 == ' ':
                    blanks += 1
        return matches, blanks

    # Calculates the emmision probability of each letter in test set against each in the training set
    def set_emmission_prob(self):
        
        total_pixels = CHARACTER_WIDTH * CHARACTER_HEIGHT
        for test_letter_index, test_letter in enumerate(self.test_letters):
            for train_letter, train_letter_grid in self.train_letters.items():
                match_score,blanks = self.compare_letters(test_letter, train_letter_grid)
                unmatch_score = total_pixels - match_score - blanks
                match_prob = (match_score + 0.0) / total_pixels
                prob = (0.75 * match_score + 0.20 * blanks + 0.05 * unmatch_score) / (CHARACTER_WIDTH * CHARACTER_HEIGHT) # 0.70 * match_prob + 0.25 * blanks + 0.05 * (1 - match_prob) is given to weigh the matches, misses and blanks in the grid comparisons. Basing this on Naive Bayes assumption of havin some prior probabailities
                if self.emission_prob[test_letter_index]:
                    if self.emission_prob[test_letter_index][train_letter]:
                        if prob != 0:
                            self.emission_prob[test_letter_index][train_letter] = -math.log(prob) 
                        else:
                            self.emission_prob[test_letter_index][train_letter] = max_val
    
    # Returns the emmision probability dictionary for the test letter for a given index
    def get_emission_probs(self, test_char):
        
        emission_prob_dict = dict()
        for char in TRAIN_LETTERS:
            if test_char in self.emission_prob and char in self.emission_prob[test_char]:
                emission_prob_dict[char] = self.emission_prob[test_char][char]
            else:
                emission_prob_dict[char] = max_val
        return emission_prob_dict

    # This get's called in in the _initi_.
    # Calculates the initial probability dictionary of starting words in a letter based on the training text file
    # Calculates the transition probability dictionary of starting words in a letter based on the training text file
    # Calls the emmission probability function to calculate the emmision probability dictionary of each letter in the test set
    def train(self, train_txt_fname):

        def normalize_dict(dict_to_normalize):
            total_log = math.log(sum(dict_to_normalize.values()))
            for key, val in dict_to_normalize.items():
                dict_to_normalize[key] = max_val if val < 1 else total_log - math.log(val)

        with open(train_txt_fname, "r") as f:
            for line in f:
                prev_letter = None
                for letter in line:
                    if letter in TRAIN_LETTERS:
                        if prev_letter is not None:
                            self.transition_prob[prev_letter][letter] += 1
                        if prev_letter == " ":
                            if letter in self.init_prob:
                                self.init_prob[letter] += 1
                            else:
                                self.init_prob[letter] = 1
                        prev_letter = letter

        normalize_dict(self.init_prob)

        for dict_to_normalize in self.transition_prob.values():
            normalize_dict(dict_to_normalize)
        
        self.set_emmission_prob()

    # Call this function to get Simple HMM (with Naive Bayes assumptions) based prediction of letters in the test picture 
    # For each test letter, get the letter from TRAIN_LETTER with the lowest emmision probability dictionary
    # Note: Optimization problem is to find minimum of the emmision probability dictionary for each test letter
    def simplified(self):
        ret = []
        for test_letter_index, test_letter in enumerate(self.test_letters):
            curr_prob = max_val
            Curr_letter = None
            for ch, prob in self.emission_prob[test_letter_index].items():
                if prob < curr_prob:
                    curr_prob = prob
                    curr_letter = ch
            ret.append(curr_letter)
        return "".join(ret)
    
    # Call this function to get Viterbi probabilities based on Naive Bayes
    # Note: Optimization problem is to find minimum of the total ((emission + transition + previous) or (emission + initial)) probability values for each test letter
    def viterbi(self):
        
        ret = []
        
        viterbi_matrix = [[None] * len(self.test_letters) for _ in range(len(TRAIN_LETTERS))]
        back_pointer = [[None] * len(self.test_letters) for _ in range(len(TRAIN_LETTERS))]
        
        for col_index in range(len(self.test_letters)):
            
            test_char_emmision_prob = self.get_emission_probs(col_index)
            
            for row_index, curr_char in enumerate(TRAIN_LETTERS):
                
                if col_index == 0:
                    init_prob = self.init_prob[curr_char] if curr_char in self.init_prob else max_val
                    final_prob = 0.0001*init_prob + test_char_emmision_prob[curr_char]
                    viterbi_matrix[row_index][col_index] = (-1, final_prob)
                else:
                    best_tuple = (-1, max_val)
                    for prev_index, prev_char in enumerate(TRAIN_LETTERS):
                        prev_prob = viterbi_matrix[prev_index][col_index - 1][1]
                        curr_prob = prev_prob + 0.001*self.transition_prob[prev_char][curr_char] + 0.999*test_char_emmision_prob[curr_char] # 0.001*transition probaility + 0.999*emission probaility to normalize the transition and emmision probailities 
                        if curr_prob < best_tuple[1]:
                            best_tuple = (prev_index, curr_prob)
                    viterbi_matrix[row_index][col_index] = best_tuple
        
        # Backtracking
        # Reference taken for this part only: https://www.geeksforgeeks.org/viterbi-algorithm-implementation/, Viterbi Class Activity, https://github.com/ssghule/Optical-Character-Recognition-using-Hidden-Markov-Models/blob/master/ocr_solver.py
        (max_index, max_prob) = (-1, max_val)

        train_rows = len(TRAIN_LETTERS)
        test_cols = len(self.test_letters)
        for row in range(train_rows):
            curr_prob = viterbi_matrix[row][test_cols - 1][1]
            if curr_prob < max_prob:
                (max_index, max_prob) = (row, curr_prob)

        for col in range(len(self.test_letters) - 1, 0, -1):
            ret.insert(0, TRAIN_LETTERS[max_index])
            max_index = viterbi_matrix[max_index][col][0]
        ret.insert(0, TRAIN_LETTERS[max_index])
        return "".join(ret)

