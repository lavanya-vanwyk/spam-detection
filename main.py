import sys
import re
import math
from naive_bayes import *

#---bayes probability algorithm
# P(x-word occuring given if in spam or ham) =  no. of emails in testing_set that have x-word + 1 /
    #                                                     total no. emails in spam/ham + 2

    # the "+1" and "+2" are laplace smoothing for if a probability is zero to avoid errors

def detect_spam(file, prior_probabilities, likelihoods, unique_words, all_words):
    '''
    Uses a trained Naive Bayes model to calculate whether a given word is spam in an email
    based on the probablities calculated in the model. The total probablity of an email being
    spam is updated with each word checked against the model data.
    '''
    with open(file, "r", encoding = "utf-8", errors = "ignore") as c_file:
        check_file = c_file.read()
        c_check_words = re.sub(r'[^a-zA-Z0-9\s]', '', check_file)
        check_list = c_check_words.lower().split()

        spamicity_hamicity = {}

        for category in prior_probabilities:
            # value to hold spam/hamicity of currrent checked word
            current_val = math.log(prior_probabilities[category])
            for word in check_list:
                if word in likelihoods[category]:
                    current_val += math.log(likelihoods[category][word])
                else: 
                  #word not in testing set
                    new_word_likelihood = 1 / (all_words + len(unique_words))
                    current_val += math.log(new_word_likelihood)

            spamicity_hamicity[category] = current_val

        if spamicity_hamicity[1] > spamicity_hamicity[0]:
            print("spam")
            return "spam"
        else:
            print("notspam")
            return "notspam"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1) 
        
    file = sys.argv[1]
    
    model_params = load_model()
    if model_params:
        prior_probabilities, likelihoods, unique_words, all_words = model_params
    else:
        # print("Starting training...")
        prior_probabilities, likelihoods, unique_words, all_words = train_naive_bayes_model()
        save_model(prior_probabilities, likelihoods, unique_words, all_words)
        
    detect_spam(file, prior_probabilities, likelihoods, unique_words, all_words)
