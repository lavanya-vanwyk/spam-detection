# Spam Detection - Exercise

This Python program takes in a text file and determines whether the contents classify it as 
a spam, or not spam email.

## Project goals and features

The aim of this program was to be able to read the contents of the text file and based on each
word, determine the level of spam, incrementing as spam words are encountered. The project constraints were to try to build this without any
external libraries, just pure Python.

## Project approach
Project restrictions: May only use built-in Python modules/packages

I decided to use a CSV file sourced from Kaggle.com that consisted of spam and 
not spam emails, and the a url to the raw CSV.

Each email message is associated with a binary label, where "1" indicates that the email is spam, and "0" indicates that it is not spam. 
Total emails in the set is **5695**. 
This is the reference link to [the dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) as retrieved from Kaggle.

I then trained a Naive Bayes classifier to grab all words in the dataset, 
(also storing a list of unique words in a set) and depending on whether the email was classified as spam or not spam, calculate the probability of the word being spam. 
This was, naturally, achieved using the Naive Bayes probability formula. This also significantly reduced the risk for false flags. 

Other implementation strategies I used were cleaning the data strings with regex. I opted not to remove stop words and omit using lemmatization (as commonly used in spam email detection) because I felt that the size of my dataset, would overcome the need for this, and it would therefore not be necessary.

## Retrospection
Most of my debugging was tweaking errors in my probability calculations, however I am proud to say that this program correctly identifies 98.0% of spam emails as spam, and 98.0% of not spam emails as not spam.
<img width="508" height="81" alt="image" src="https://github.com/user-attachments/assets/d3618b8e-30cb-4ef7-87a5-f3352c980063" />

## What's next

I plan to update the classifier to allow the user to choose between spam detection or sentiment analysis. 

## Running instructions

Clone the repository:

`$ git clone https://github.com/lavanya-vanwyk/spam-detection

Enter the project directory:

`cd spam-detection`

Open spam.py in your IDE of choice. Your device must be connected to the internet on the first run in order to build the model from the url.
