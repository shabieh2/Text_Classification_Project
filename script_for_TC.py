from sklearn.datasets import fetch_20newsgroups
# MODULE IMPORTS

import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


train=fetch_20newsgroups(subset='train',shuffle=True)
test=fetch_20newsgroups(subset='test',shuffle=True)


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

lemma=WordNetLemmatizer()

class LemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (lemma.lemmatize(w) for w in analyzer(doc))



#Tokenization

vectorizer = LemmedTfidfVectorizer(sublinear_tf=True, max_df=0.6,
                                 stop_words='english',ngram_range=(1,2))


import time
start_time=time.time()

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SGD= SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=50, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)

SGDE=SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=50, n_jobs=1,
       penalty='elasticnet', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
       
pipe_svm= Pipeline([('vectorizer',vectorizer),('svm',SGD)])
predicted=pipe_svm.fit(train.data,train.target).predict(test.data)
print (np.mean(predicted==test.target),(time.time()-start_time))

import time
start_time=time.time()

from sklearn.model_selection import GridSearchCV

parameters= {'vectorizer__max_df':(0.7,0.8)}

gs_svm=GridSearchCV(pipe_svm,parameters,n_jobs=1,verbose=1)

gs_svm= gs_svm.fit(train.data,train.target)

print("%s seconds" % (time.time()-start_time))

gs_svm.best_params_

gs_svm.best_score_
    
gs_svm.cv_results_
