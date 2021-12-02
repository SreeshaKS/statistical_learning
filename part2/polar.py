#!/usr/local/bin/python3
#
# Authors: [PLEASE PUT YOUR NAMES AND USER IDS HERE]
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np


# calculate "Edge strength map" of an image
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return sqrt(filtered_y ** 2)


# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range(int(max(y - int(thickness / 2), 0)), int(min(y + int(thickness / 2), image.size[1] - 1))):
            image.putpixel((x, t), color)
    return image


def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [(pt[0] + dx, pt[1] + dy) for dx in range(-3, 4) for dy in range(-2, 3) if
                   dx == 0 or dy == 0 or abs(dx) == abs(dy)]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    imageio.imwrite('test.png', new_image)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)


def hmm(edge_strength_matrix, p_transition):
    # viterbi implementation
    col_size = edge_strength_matrix.shape[1]
    row_size = edge_strength_matrix.shape[0]
    viterbi = zeros(col_size)
    col_sums = zeros(col_size)
    p_state = zeros((row_size, col_size))
    back_pointer = zeros((row_size, col_size))

    for col in range(col_size):
        for row in range(row_size):
            col_sums[col] += edge_strength_matrix[row][col]
    s = sum(col_sums)
    # initialise probabilities
    for row in range(row_size):
        # initial probability * observation probability
        # p_state[row][0] = edge_strength_matrix[row][0] * (edge_strength_matrix[row][0] / col_sums[0])
        p_state[row][0] = edge_strength_matrix[row][0] / s

    # calculating state probabilities of each node using viterbi
    # for col in range(1, col_size):
    #     for row in range(row_size):
    #         p_maximum = 0
    #         for j in range(-4, 5):
    #             if (row + j < row_size) and (row + j >= 0):
    #                 if p_maximum < p_state[row + j][col - 1] * p_transition[abs(j)]:
    #                     p_maximum = p_state[row + j][col - 1] * p_transition[abs(j)]
    #                     back_pointer[row][col] = row + j
    #                 p_state[row][col] = (edge_strength_matrix[row][col] / 100) * p_maximum
    p_state, back_pointer = viterbi_recur(
        (1, col_size, 1),
        (0, row_size, 1),
        col_size, row_size,
        p_state,
        back_pointer,
        edge_strength_matrix,
        p_transition
    )
    viterbi = back_track(p_state, col_size, viterbi, back_pointer)
    # p_max_index = argmax(p_state[:, col_size - 1])
    #
    # for col in range(col_size - 1, -1, -1):
    #     viterbi[col] = int(p_max_index)
    #     p_max_index = back_pointer[int(p_max_index)][col]

    return viterbi, p_state, back_pointer


def back_track(p_state, col_size, viterbi, back_pointer):
    p_max_index = argmax(p_state[:, col_size - 1])
    for col in range(col_size - 1, -1, -1):
        viterbi[col] = int(p_max_index)
        p_max_index = back_pointer[int(p_max_index)][col]
    return viterbi


def viterbi_recur(col_loop, row_loop, col_size, row_size, p_state, back_pointer, edge_strength_matrix,
                  p_transition_offset):
    col_start, col_stop, col_step = col_loop
    row_start, row_stop, row_step = row_loop

    for col in range(col_start, col_stop, col_step):
        for row in range(row_start, row_stop, row_step):
            p_maximum = 0
            l = len(p_transition_offset)
            for j in range(-(l - 1), l):
                if (row + j < row_size) and (row + j >= 0):
                    if p_maximum < p_state[row + j][col - 1] * p_transition_offset[abs(j)]:
                        p_maximum = p_state[row + j][col - 1] * p_transition_offset[abs(j)]
                        back_pointer[row][col] = row + j
                    p_state[row][col] = (edge_strength_matrix[row][col] / 100) * p_maximum
    # for col in range(col_start, col_stop, col_step):
    #     for row in range(row_start, row_stop, row_step):
    #         p_maximum = 0
    #         for j in range(-2, 3):
    #             if (row + j < row_size) and (row + j >= 0):
    #                 if p_maximum < p_state[row + j][col - 1] * p_transition_offset[abs(j)]:
    #                     p_maximum = p_state[row + j][col - 1] * p_transition_offset[abs(j)]
    #                     back_pointer[row][col] = row + j
    #                 p_state[row][col] = (edge_strength_matrix[row][col] / 100) * p_maximum
    return p_state, back_pointer


# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception(
            "Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [int(i) for i in sys.argv[2:4]]
    gt_icerock = [int(i) for i in sys.argv[4:6]]

    # load in image
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength_matrix = edge_strength(input_image)
    imageio.imwrite('edges.png', uint8(255 * edge_strength_matrix / (amax(edge_strength_matrix))))
    import pdb

    col_size = edge_strength_matrix.shape[1]
    row_size = edge_strength_matrix.shape[0]
    # ********************** Simplified ***************************************************
    airice_simple = argmax(edge_strength_matrix, axis=0)
    edge_strength_matrix = edge_strength_matrix.tolist()

    icerock_simple = []
    for idx, row in enumerate(airice_simple):
        i = 0
        r = []
        while i < 10:
            edge_strength_matrix[row - i][idx] = -1
            i += 1

    icerock_simple = argmax(edge_strength_matrix, axis=0)
    # ********************** Simplified ***************************************************

    # ********************** Viterbi-AirIce ***************************************************

    edge_strength_matrix = edge_strength(input_image)
    # p_transition_offset = [0.5, 0.4, 0.1, 0.5, 0.005]
    p_transition_offset = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0005, 0.00005, 0]
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    mu, sigma = 0, 0.1  # mean and standard deviation
    s = np.random.normal(mu, sigma, 5)

    airice_hmm, airice_p_state, airice_back_pointer = hmm(edge_strength_matrix, p_transition_offset)

    # ********************** HumanFeedback - Viterbi-AirIce ***************************************************
    # tweak probability distribution
    for row in range(row_size):
        airice_p_state[row][gt_airice[0]] = 0
    airice_p_state[gt_airice[1]][gt_airice[0]] = 1
    # update forward
    p_state, airice_back_pointer = viterbi_recur(
        (gt_airice[0] - 1, 0, -1),
        (gt_airice[1] - 1, -1, -1),
        col_size, row_size,
        airice_p_state,
        airice_back_pointer,
        edge_strength_matrix,
        p_transition_offset
    )
    # update backward
    p_state, airice_back_pointer = viterbi_recur(
        (gt_airice[0] + 1, col_size, 1),
        (0, row_size, 1),
        col_size, row_size,
        airice_p_state,
        airice_back_pointer,
        edge_strength_matrix,
        p_transition_offset
    )
    airice_feedback = back_track(p_state, col_size, airice_hmm, airice_back_pointer)
    # ********************** HumanFeedback - Viterbi-AirIce ***************************************************
    # ********************** Viterbi-AirIce ***************************************************

    # ********************** Viterbi-IceRock ***************************************************
    # remove air-ice boundary
    for index, y_coord in enumerate(airice_hmm):
        i = 0
        while i < 10:
            edge_strength_matrix[int(y_coord) - i][index] = 0
            edge_strength_matrix[int(y_coord) + i][index] = 0
            i += 1

    icerock_hmm, icerock_p_state, icerock_back_pointer = hmm(edge_strength_matrix, p_transition_offset)
    # ********************** HumanFeedback - Viterbi-IceRock ***************************************************
    # tweak probability distribution
    for row in range(row_size):
        icerock_p_state[row][gt_icerock[0]] = 0
    icerock_p_state[gt_icerock[1]][gt_icerock[0]] = 1
    # update forward
    icerock_p_state, icerock_back_pointer = viterbi_recur(
        (gt_icerock[0] - 1, 0, -1),
        (gt_icerock[1] - 1, -1, -1),
        col_size, row_size,
        icerock_p_state,
        icerock_back_pointer,
        edge_strength_matrix,
        p_transition_offset
    )
    # update backward
    icerock_p_state, icerock_back_pointer = viterbi_recur(
        (gt_icerock[0] + 1, col_size, 1),
        (0, row_size, 1),
        col_size, row_size,
        icerock_p_state,
        icerock_back_pointer,
        edge_strength_matrix,
        p_transition_offset
    )
    icerock_feedback = back_track(icerock_p_state, col_size, icerock_hmm, icerock_back_pointer)
    # ********************** HumanFeedback - Viterbi-IceRock ***************************************************
    # ********************** Viterbi-IceRock ***************************************************

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
    # airice_simple = [image_array.shape[0] * 0.25] * image_array.shape[1]
    # airice_hmm = [image_array.shape[0] * 0.5] * image_array.shape[1]
    # airice_feedback = [image_array.shape[0] * 0.75] * image_array.shape[1]
    # icerock_simple = [image_array.shape[0] * 0.25] * image_array.shape[1]
    # icerock_hmm = [image_array.shape[0] * 0.5] * image_array.shape[1]
    # icerock_feedback = [image_array.shape[0] * 0.75] * image_array.shape[1]

    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
