# Spam Email Classifier

This project checks if a message is spam or not using Naive Bayes.

Libraries used: pandas, scikit-learn

Steps:
- load the SMS spam dataset
- convert the text into numbers using CountVectorizer
- train a MultinomialNB model
- check accuracy on test data

To run:
pip install -r requirements.txt
python spam_classifier.py
