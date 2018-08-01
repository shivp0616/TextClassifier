# TextClassifier
This repository contains different machine learning models to classify text documents using supervised machine learning approach.
The models that we have used here are Support Vector Machine, Random Forest Classifier and Multinomial Naive Bayes.

Result(accuracy in %) using default parameters: 

|               |SVM     | RFC   |MNB   |
| ------------- |-------:| -----:|-----:|
| Training      | 99     | 99    |93    |
| Testing       | 84     | 64    | 77   |


The dataset that is used is 20NewsGroups. It was originally collected by Ken Lang.
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups
The same can be downloaded from http://qwone.com/~jason/20Newsgroups/. I used the "bydate" version since cross-experiment comparison is easier (no randomness in train/test set selection), newsgroup-identifying information has been removed and it's more realistic because the train and test sets are separated in time.
The data is organized into 20 different newsgroups, each corresponding to a different topic.
Some of the newsgroups are very closely related to each other (e.g. comp.sys.ibm.pc.hardware / comp.sys.mac.hardware), while others are highly unrelated (e.g misc.forsale / soc.religion.christian).

Here is a list of the 20 newsgroups, partitioned (more or less) according to subject matter: 
comp.graphics
comp.os.ms-windows.misc
comp.sys.ibm.pc.hardware
comp.sys.mac.hardware
comp.windows.x
rec.autos
rec.motorcycles
rec.sport.baseball
rec.sport.hockey
sci.crypt
sci.electronics
sci.med
sci.space
misc.forsale
talk.politics.misc
talk.politics.guns
talk.politics.mideast
talk.religion.misc
alt.atheism
soc.religion.christian

The structure of the dataset is like:

20news-bydate-train/
|-- alt.atheism
|   |-- 49960
|   |-- 51060
|   |-- 51119

|-- comp.graphics
|   |-- 37261
|   |-- 37913
|   |-- 37914
|   |-- 37915
|   |-- 37916
|   |-- 37917
|   |-- 37918
|-- comp.os.ms-windows.misc
|   |-- 10000
|   |-- 10001
|   |-- 10002
|   |-- 10003
|   |-- 10004
|   |-- 10005 

If you want to work on your own data, all you have to do is to replace the folder's data with your own data.
The name of each folder will be treated as labels.
