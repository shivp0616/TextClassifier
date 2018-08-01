"""
@author: shivprakash1997
"""

import numpy as np
from sklearn.datasets import load_files

# importing the datasets
twenty_train = load_files('20news-bydate-train', encoding = "ISO-8859-1")
twenty_test = load_files('20news-bydate-test', encoding = "ISO-8859-1")

# see the targets
print(twenty_train.target_names)

# You can either compile the next three steps one by one or you can compile them in one go using pipeline 

#(Step 1) Turn the text documents into vectors of word frequencies
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(twenty_train.data)
y_train = twenty_train.target

#(Step 2) Implementing the TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)

#(Step 3) Using MNB to train the model
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB().fit(X_train_tfidf, y_train)
print("Training score: {0:.1f}%".format(
    classifier.score(X_train, y_train) * 100))

# using pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


# Performance of NB Classifier
#on training dataset
predicted = text_clf.predict(twenty_train.data)
np.mean(predicted == twenty_train.target)

#on testing dataset
predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)

#test custom text
testing = ["Under typical circumstances, Texas Rangers starter Cole Hamels might be the prize of the trade deadline. He's a tested veteran with four All-Star appearances and a 2008 World Series MVP Award on his resume."]
predicted_MNB = text_clf.predict(testing)
