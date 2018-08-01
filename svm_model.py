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

#(Step 3) Using SVM to train the models
from sklearn.linear_model import SGDClassifier

text_clf_svm = SGDClassifier().fit(X_train_tfidf, y_train)
print("Training score: {0:.1f}%".format(
    text_clf_svm.score(X_train, y_train) * 100))


#Using pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', SGDClassifier())])
text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)

# Performance of SV Classifier
#on training dataset
predicted = text_clf_svm.predict(twenty_train.data)
np.mean(predicted == twenty_train.target)

#on testing dataset
predicted = text_clf_svm.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)

#Testing custom data
testing = ["Well i switched my bike on and took my hockey"]
print(twenty_train.target_names[text_clf_svm.predict(testing)[0]])
