# <div align="center"> CS B551 - Assignment 3: Probability and Statistical Learning
####  <div align="center"> CSCI B551 - Elements of Artificial Intelligence

<br>

###### Name: Arunima Shukla, Sreesha Srinivasan Kuruvadi  , Adit Manurkar
###### Email: *arushuk@iu.edu, sskuruva@iu.edu, amanurk@iu.edu*
<br>

***
### [Part 1: Part-of-speech tagging](https://github.iu.edu/cs-b551-fa2021/amanurk-arushuk-sskuruva-a3/tree/master/part1)
***

#### Problem Statement:

A basic problems in Natural Language Processing is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.). Sometimes this is easy: a sentence like “Blueberries are blue” clearly consists of a noun, verb, and adjective, since each of these words has only one possible part of speech (e.g., “blueberries” is a noun but can’t be a verb).
Your goal in this part is to implement part-of-speech tagging in Python, using Bayes networks.

#### Approach
Goal of this problem is to mark each word in the sentence with its part of speech. The given training data has 12 parts of speeches, and they are followed by the word in the sentence. There are three different Baye's net model that needs to be modeled using training data and accuracy for the not seen data has to be predicted using these models.
Using Training data, we found the initial probabilities, emission probability P(wi | si) and transition probabilities p(Si | Si-1). We also stored the part of speech tags and words in an dictianary along with part of speech probabilities. Probabilities for words which are not observed in the training set are set to be 1/10^10.

- Simplified model: This model uses basics of Bayes Nets. Each word in the sentence is associated to only one part of speech tag. and select the best among them. THe max probability of the word given the part of speech is multiplied by part of speech probability to get the final results.
- For the Viterbi model: In this model, words have more than one independence to each other than the simple word to pos dependency. We used the maximum probability till each word in the sentence. and also the max proability in the state before which are pre-calculated.
- Posterior: Simple: Calculated logs of emission probabilities and returned the sum HMM Viterbi: Calculated logs of emission probabilities and transition probabilities Complex:
- MCMC - Gibbs sampling - Posterior Probabilities of complex model as P = p(word/pos)*p(pos/prev_pos)*p(next_pos/pos)
#### Assumptions 
For all the models, if we encounter a new word, emission probability was set to be 1/10^10
***
### [Part 2: Ice tracking](https://github.iu.edu/cs-b551-fa2021/amanurk-arushuk-sskuruva-a3/tree/master/part2)

***

#### Problem Statement:
In this problem, we’ll help solve global warming. :)

#### Goal:
To understand how rising global temperatures affect ice at the Earth’s north and south poles, glaciologists need information about the structure of the ice sheets. The traditional way of doing this is to drill into the ice and remove an ice core. But a single ice core can take many months to drill, and only gives information about the ice at a single latitude-longitude point. To expedite this process, scientists have developed radar systems that allow an airplane to collect an approximate “cross section” of the ice below the airplane’s flight path 
One is the very dark line near the top, which is the boundary between the air and the ice. There’s also a deeper line which shows the boundary between the ice and the bedrock.
In this part, we’ll create code to try to find these two boundaries (air-ice and ice-rock). We’ll make some assumptions to make this possible.

#### Assumptions:
First, you can assume that the air-ice boundary is always above the ice-rock boundary by a significant margin (say, 10 pixels). Second, you can assume that the two boundaries span the entire width of the image. Taken together these, two assumptions mean that in each column of the image, there is exactly one air-ice boundary and exactly one ice-rock boundary, and the ice-rock boundary is always below. Third, we assume that each boundary is relatively “smooth” — that is, a boundary’s row in one column will be similar in the next column. Finally, you can assume that the pixels along the boundaries are generally dark and along a strong image edge (sharp change in pixel values)..

#### Command:
<code>  python3 ./polar.py input_file.jpg airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord
 </code>

#### Approach
##### 1. In the simple model, for air-ice we used numpy library to find the maximum row index in every column using the numpy.argmax() function.
And for ice-rock boundary -
- Set air ice boundary edge strength to zero.
- numpy library to find the maximum row index in every column using the numpy.argmax() function.
##### 2. For the HMM-viterbi model, to populate the transition probabilities, we used viterbi algorithm with dynamic programming and back tracking method.

##### 3. When additional human feedback is given we just tweaked the previous logic, made that particular pixel's state probability maximum, so the resultant path will be the path that passes through the human labeled pixel always. 
- set the p_distribution corresponding to the pixel to 1
- recur right
- recur left
- update max probability coordinates through backward algorithm
- repeat the same for ice-rock boundary

##### 4. Transition offset probabilities are used to decrease the probability of pixels far away from the current row
- Picked the below two values
  - trans_1 = [0.5, 0.4, 0.1] - causes aggressive decrease in probabilities
  - trans_2 = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0005, 0.00005, 0] - close to a Guassian distribution.
  - trans_2 works better than trans1 as it considers more possibilities on the two sides of the row and less aggressive nearer to the row and exponentially aggressive farther away from the row.


###### Original Image
![Original Image](https://github.com/SreeshaKS/statistical_learning/blob/master/part2/test_images/23.png)

<code>python3 ./polar.py test_images/23.png 147 35 148 68</code>

###### Using - [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0005, 0.00005, 0]
![Air-ice boundary](https://github.com/SreeshaKS/statistical_learning/blob/master/part2/air_ice_output.png)
![Ice-rock boundary](https://github.com/SreeshaKS/statistical_learning/blob/master/part2/ice_rock_output.png)

##### Using - [0.5, 0.4, 0.1]
![Air-ice boundary](https://github.com/SreeshaKS/statistical_learning/blob/master/part2/air_ice_output.png)
![Ice-rock boundary](https://github.com/SreeshaKS/statistical_learning/blob/master/part2/ice_rock_output_low_transition.png)

#### References:
##### 1. http://www.cs.cmu.edu/~guestrin/Class/10701/slides/hmms-structurelearn.pdf
##### 4. https://web.stanford.edu/~jurafsky/slp3/A.pdf
##### 5. Canvas lecture slides

### [Part 3: Reading text](https://github.com/SreeshaKS/statistical_learning/tree/master/part3)


#### Problem Statement
To show the versatility of HMMs, let’s try applying them to another problem; if you’re careful and you plan ahead, you can probably re-use much of your code from Parts 1 and 2 to solve this problem. 
Our goal is to recognize text in an image – e.g., to recognize that Figure 2 says “It is so ordered.” But the images are noisy, so any particular letter may be difficult to recognize. However, if we make the assumption that these images have English words and sentences, we can use statistical properties of the language to resolve ambiguities, just like in Part 2.

#### Approach
Naive Bayes Assumption based probabilities are the key to good Image Character Recognition. The emission probabilities without the Naive Bayes assumption works well only for clean images of characters. However, our code works significantly well even for the images with lot of noise in addition to clear character pixel values being present. This happens when we only match the grids pixels which match in the test and train images. Having defined the probabilities to the "unmatched" factor greatly improves recognition of images with sparse pixels (eg : test-16.png test-8.png).

We have also factored in the blanks or the " " pixels separately with a minimum probability factor as it acts like a string cleaning mechanism, considering the blank spaces in the image at the places which do not outline a character, can help us to detect a character in case of sparsely pixelated image. Tweaking the weight values given to match, unmatch, and blanks is the key to tradeoff between making the model better for detecting the noisy, or sparse, images.

The current values of weights is better at detecting sparsely pixelated images but can give some wrong predictions for the noisy images. But this can be varied by changing weights values on the line # 61 of char_recognizer :
''' prob = (0.70 * match_score + 0.25 * blanks + 0.05 * unwatch_score) / (CHARACTER_WIDTH * CHARACTER_HEIGHT) # 0.75 * match_prob + 0.20 * blanks + 0.05 * (1 - match_prob) is given to weigh the matches, misses and blanks in the grid comparisons. Basing this on Naive Bayes assumption of having some prior probabilities'''. These values can be tweaked for testing too
Viterbi algorithm really helps in improving the probabilities of proper words. For eg: UN1TED predicted by the HMM gets converted to UNITED when run with Viterbi.
We are able to predict almost humanly unreadable images, with considerable accuracy. However, some images (eg. test-15.png) do give unwanted prediction because of the transitional probability consideration. For completeness, the essence of emmision probability in this use case if the probability of matching of test image character with each training image characters (given in courier.png say). The transition probability is the probability of occurance of one letter after a specific letter calculated based on the text training set.
