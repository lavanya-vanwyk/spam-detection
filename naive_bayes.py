import urllib.request
import os
import pickle
import io
import csv
import re

model_file = "naive_bayes_model.pkl"
def train_naive_bayes_model():
    '''
    Trains a machine learning model off of a data set of over 5,000 emails.
    Add the data to a list as tuple along with either a 0 or 1, denoting whether it is spam 
    or ham.
    
    This function returns the prior probabilities for spam and ham, 
    likelihoods, a list of all the words (non repeating), and the total words across all emails in
    the data. 
    
    '''
    url = "https://raw.githubusercontent.com/lavanya-vanwyk/spam-detection-data/97e84dfd335eb39ab8a2232ef91b1f3a98f74263/emails.csv"
    response = urllib.request.urlopen(url)
    data = response.read()
    data_as_str = data.decode("utf-8")
    csv_file = io.StringIO(data_as_str)
    training_data = csv.reader(csv_file)

    training_set = []
    counts = {0: 0, 1: 0}
    total_word_counts = {0: {}, 1: {}}
    unique_words = set()
    all_words = 0

    for line in training_data:
        text = line[0]
        category = line[1]
        training_set.append((text, int(category)))
        
    for (text, category) in training_set:
        counts[category] += 1
        cleaned_email = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        email_words = cleaned_email.lower().split()

        for w in email_words:
            unique_words.add(w)
             # increment count for the number of times it appears
            total_word_counts[category][w] = total_word_counts[category].get(w, 0) + 1

    all_words += sum(total_word_counts[category].values()) 
        
    total_emails = counts[0] + counts [1]

    spam_probability = counts[1] / total_emails # 0.7
    ham_probability = counts[0] / total_emails # 0.3

    prior_probabilities = {0: ham_probability, 1: spam_probability}
    likelihoods = {0:{}, 1:{}}
    
    for category in total_word_counts:
        #spam ham RESPECTIVE
        category_count = sum(total_word_counts[category].values())
        bayes_denominator = category_count + len(unique_words)

        for w in unique_words:
           words_in_category = total_word_counts[category].get(w, 0)
           bayes_numerator = words_in_category + 1
           likelihoods[category][w] = bayes_numerator / bayes_denominator
    
    return prior_probabilities, likelihoods, unique_words, all_words

def save_model(prior_probabilities, likelihoods, unique_words, all_words):
    '''
    This function ensures that when the program is first run, the trained model is saved.
    This avoids any issues with performance in having to retrain the model each time an
    email needs to be checked.
    '''
    model_data = {
        'prior_probabilities': prior_probabilities,
        'likelihoods': likelihoods,
        'unique_words': unique_words,
        'all_words': all_words
    }
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)

def load_model():
    for file in os.listdir("."):
        if file.endswith(".pkl"):
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            return (
                model_data['prior_probabilities'], 
                model_data['likelihoods'], 
                model_data['unique_words'], 
                model_data['all_words']
        )
        else:
            return None
