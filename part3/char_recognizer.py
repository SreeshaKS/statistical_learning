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
        
        self.init_prob = dict()

        self.emission_prob = defaultdict()
        for i in range(len(test_letters)):
            self.emission_prob[i] = dict()
            for j in TRAIN_LETTERS:
                self.emission_prob[i][j] = max_val

        self.transition_prob = defaultdict()
        for i in TRAIN_LETTERS:
            self.transition_prob[i] = dict()
            for j in TRAIN_LETTERS:
                self.transition_prob[i][j] = min_val
        self.train(train_txt_fname)

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

    def set_emmission_prob(self):
        
        total_pixels = CHARACTER_WIDTH * CHARACTER_HEIGHT
        for test_letter_index, test_letter in enumerate(self.test_letters):
            for train_letter, train_letter_grid in self.train_letters.items():
                match_score,blanks = self.compare_letters(test_letter, train_letter_grid)
                unmatch_score = total_pixels - match_score - blanks
                match_prob = (match_score + 0.0) / total_pixels
                prob = (0.70 * match_score + 0.25 * blanks + 0.05 * unmatch_score) / (CHARACTER_WIDTH * CHARACTER_HEIGHT)
                if self.emission_prob[test_letter_index]:
                    if self.emission_prob[test_letter_index][train_letter]:
                        if prob != 0:
                            self.emission_prob[test_letter_index][train_letter] = -math.log(prob)
                        else:
                            self.emission_prob[test_letter_index][train_letter] = max_val
    
    def get_emission_probs(self, noisy_char):
        
        emission_prob_dict = dict()
        for char in TRAIN_LETTERS:
            if noisy_char in self.emission_prob and char in self.emission_prob[noisy_char]:
                emission_prob_dict[char] = self.emission_prob[noisy_char][char]
            else:
                emission_prob_dict[char] = max_val
        return emission_prob_dict

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
            
    def viterbi(self):
        ret = []
        rows = len(TRAIN_LETTERS)
        cols = len(self.test_letters)
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
                        curr_prob = prev_prob + 0.001*self.transition_prob[prev_char][curr_char] + 0.999*test_char_emmision_prob[curr_char]
                        if curr_prob < best_tuple[1]:
                            best_tuple = (prev_index, curr_prob)
                    viterbi_matrix[row_index][col_index] = best_tuple
        
        (max_index, max_prob) = (-1, max_val)
        for row in range(rows):
            curr_prob = viterbi_matrix[row][cols - 1][1]
            if curr_prob < max_prob:
                (max_index, max_prob) = (row, curr_prob)

        for col in range(len(self.test_letters) - 1, 0, -1):
            ret.insert(0, TRAIN_LETTERS[max_index])
            max_index = viterbi_matrix[max_index][col][0]
        ret.insert(0, TRAIN_LETTERS[max_index])
        return "".join(ret)

