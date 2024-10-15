from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

class SVM():

    def __init__(self):
        self.name = "20NewsGroup"
        self.lb = [1e-2, 1e-2]
        self.ub = [1e3, 1e3]
        self.domain = np.array([[1e-2, 1e-2],
                         [1e3, 1e3]])
        self.D = 2
        self.opt_y = -1

    def obj(self, x):

        twenty_dataset = fetch_20newsgroups(
            shuffle=True,
        )

        # model parameters in range  [10−2, 10^3]
        C = x[0]
        gamma = x[1]

        # build classifier as a Pipeline
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC(C=C, gamma=gamma)),
        ])


        average_cross_validation_score = cross_val_score(text_clf, twenty_dataset.data, twenty_dataset.target, cv=5).mean()

        return -average_cross_validation_score
