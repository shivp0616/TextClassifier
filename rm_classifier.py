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

#(Step 3) Using Random Forest Classifier to train the model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_tfidf, y_train)
predicted = rf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)
print("Testing score: {0:.1f}%".format(
    rf.score(twenty_test.data, twenty_test.target) * 100))

#Using Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

text_clf_rfc = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),('clf', RandomForestClassifier())])
text_clf_rfc = text_clf_rfc.fit(twenty_train.data, twenty_train.target)


# Performance of RF Classifier
#on training dataset
predicted = text_clf_rfc.predict(twenty_train.data)
np.mean(predicted == twenty_train.target)

#on testing dataset
predicted = text_clf_rfc.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)

#Testing custom data
testing = ["Well i switched it on and suddenly there was a huge sound and my mac was no more"]
print(twenty_train.target_names[text_clf_svm.predict(testing)[0]])
