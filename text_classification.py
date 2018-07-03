import nltk # please download wordnet and stopwords if not already downloaded 

import pandas as pd
import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


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
from sklearn.decomposition import NMF
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import sklearn



# Import data and perform train-test split

a=pd.read_csv("C:/Users/shabieh2/Desktop/data_train.csv",encoding='latin1')

b=pd.read_csv("C:/Users/shabieh2/Desktop/data_dev.csv",encoding='latin1')

c=pd.read_csv("C:/Users/shabieh2/Desktop/data_eval.csv",encoding='latin1')




text=a['text']
label=a['label']

text_dev=b['text']
label_dev=b['label']

text_eval=c['text']



# Stemming and Lemmatization


stemmer = SnowballStemmer("english", ignore_stopwords=True)
lemmer=LancasterStemmer()
lemma=WordNetLemmatizer()

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: ([lemma.lemmatize(w) for w in analyzer(doc)])        
    
SCV = StemmedCountVectorizer(ngram_range=(1,3))

LCV=LemmaCountVectorizer(ngram_range=(1,3))


#SVM

text_svm_stemmed = Pipeline([('vectorizer', LCV), ('tfidf', TfidfTransformer(use_idf=True)), 
                             ('svm', SGDClassifier(penalty='elasticnet', l1_ratio=0.1, alpha=0.00001))])


text_svm_stemmed.fit(text,label)    

predicted=text_svm_stemmed.predict(text_dev)

np.mean(predicted==label_dev)

sklearn.metrics.f1_score(label2,predicted,average='weighted')


# Use GridSearch to extract optimal params 

parameters = {'vectorizer__ngram_range': [(1, 3)],
               
               'svm__l1_ratio': (0.1,0.15),}

gs_clf = GridSearchCV(text_svm_stemmed, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(text,label)

gs_clf.best_score_
gs_clf.best_params_

#Check crossval score

cross_val_score(text_svm_stemmed,text,label, cv=10)

# Add the label to data_Eval

label_eval=text_svm_stemmed.predict(text_eval)

c['label']=label_eval

c=c.drop(['text'],axis=1)

c.to_csv("C:/Users/shabieh2/Desktop/data_eval_labeled.csv",index=False)