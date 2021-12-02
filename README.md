# amanurk-arushuk-sskuruva-a3
a3 created for amanurk-arushuk-sskuruva

## PART 3:
- Naive Bayes Assumption based probabilities are the key to good Image Character Recognition. The emmission probabilities with the assumption works well only for clean images of charcters. However, this works significantly well even for the the images with lot of noice in addition to clear character pixel values being present. This happens when we only match the grids pixels which match in the test and train images. Having defined the probabilities to the "unmatched" factor greatly improves recognition of images with sparse pixels (eg : test-16.png test-8.png). 

- We have also factored in the blanks or the " " pixels sepeately with a minimum probability factor as it acts like a string cleaning mechanism, considering the blank spaces in the image at the places which do not outline a character, can help us detecting a character in case of sparsely pixeled image. Tweaking the weight values given to match, unmatch, and blanks is the key to tradeoff between making the model better for detecting the noisy, or sparse, images.

- The current values of weights is better at detecting sparsely pixelated images but can give some wrong predictions for the noisy images. But this can be varied by changing weights values on the line # 61 of char_recognizer ''' prob = (0.70 * match_score + 0.25 * blanks + 0.05 * unmatch_score) / (CHARACTER_WIDTH * CHARACTER_HEIGHT) # 0.70 * match_prob + 0.25 * blanks + 0.05 * (1 - match_prob) is given to weigh the matches, misses and blanks in the grid comparisons. Basing this on Naive Bayes assumption of havin some prior probabailities'''. These values can be tweaked for testing too

- Viterbi algorithm really helps in improving the probabalities of proper words. For eg: UN1TED prideicted by the HMM get's converted to UNITED when run with Viterbi.
We are able to predict almost humanly unreadable images, with considerable accuracy. However some images (eg. test-15.png) do give unwanted prediction because of the transitional probability consideration
