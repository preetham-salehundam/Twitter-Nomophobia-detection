"""
@author: Preetham Salehundam
@email: salehundam.2@wright.edu
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

import string
from spacy.lang.en.stop_words import  STOP_WORDS
from spacy.lang.en import English
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics
import numpy as np
#using smote to deal imbalance
from imblearn.over_sampling import SMOTE
punctuations = string.punctuation

# stop words
nlp = spacy.load("en")
stop_words = STOP_WORDS
parser = English()
def tokenizer(sentence, lemmatize=True, remove_stops=True):
    """
    :param sentence:
    :param lemmatize:
    :param remove_stops:
    :return:
    """
    my_tokens = parser(sentence)
    #lemmetization
    if lemmatize:
        my_tokens = [word.lemma_.lower().strip() for word in my_tokens if word.lemma_ !="-PRON-" and "http" not in word.lemma_ and "@" not in word.lemma_]
    #remove stop words
    if remove_stops:
        my_tokens = [word for word in my_tokens if word not in STOP_WORDS and word not in punctuations]
    return my_tokens

# def filter_tokens(tokens, filter):
#     return [token for token in tokens if token not in filter]


class predictors(TransformerMixin):
    def __init__(self, smote):
        self.smote = smote
        self.resampled=None

    def transform(self, X, **transform_params):
        return self.resampled

    def fit(self, X, y=None, **fit_params):
        self.resampled = self.smote.fit_resample(X,y)
        return self

    def get_params(self):
        return {}

def clean_text(text):

    return text.strip().lower()

def get_n_imp_features(classifier, feature_names, n =10):
    if isinstance(classifier, MultinomialNB):
        neg_class_prob_sorted = classifier.feature_log_prob_[0, :].argsort()[::-1]
        pos_class_prob_sorted = classifier.feature_log_prob_[1, :].argsort()[::-1]
        pos = np.take(feature_names, pos_class_prob_sorted[:n])
        neg = np.take(feature_names, neg_class_prob_sorted[:n])
        return pos, neg
    if isinstance(classifier, LogisticRegression):
        feature_imps = classifier.coef_[0].argsort()[::-1]
        imp_features = np.take(feature_names, feature_imps[:n])
        return imp_features, None
    if isinstance(classifier, DecisionTreeClassifier):
        feature_imps = classifier.feature_importances_.argsort()[::-1]
        imp_features = np.take(feature_names, feature_imps[:n])
        return  imp_features, None


if __name__ == "__main__":

    df_tweets = pd.read_csv("project_8_labels_Wed_Nov_13_2019.csv")

    df_tweets["Label"][(df_tweets["Label"] < 5)] = 1
    df_tweets["Label"][(df_tweets["Label"] >= 5)] = 0
    df_tweets_0 = df_tweets[df_tweets["Label"] == 0].sample(n=72)
    df_tweets_0.head(10).to_csv("neg.csv")
    df_tweets_1 = df_tweets[df_tweets["Label"] == 1].sample(n=72)
    df_tweets_1.head(10).to_csv("pos.csv")
    df_tweets = df_tweets_0.append(df_tweets_1, ignore_index = True)
    df_tweets["Text"].apply(tokenizer)
    bow_vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(3,5))
    tfidf_vector = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,3))
    # nlp = spacy.load("en_trf_bertbaseuncased_lg")
    # doc = nlp("Apple Inc shares rose on the news. Apple watch is awesome.")
    # print(doc[0].similarity(doc[8]))
    # print(len(doc))
    # print(doc._.trf_word_pieces_)
    # print(doc._.trf_last_hidden_state.shape)
    # for key in doc._:
    #     #     print(key)
    X = df_tweets["Text"]
    y = df_tweets["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    accuracy=[]
    Precision=[]
    Recall=[]
    f1_scores=[]
    imps=[]
    imps_a = []
    imps_b = []

    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train, test in cv.split(X,y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_test, return_counts=True))

        classifier = LogisticRegression(class_weight='balanced')
        classifier_NB = MultinomialNB()
        classifier_tree= DecisionTreeClassifier(criterion="gini")



        pipeline = Pipeline(steps=[("vectorizer", tfidf_vector),("classifier",classifier_tree)])


        pipeline.fit(X_train, y_train)

        predicted=pipeline.predict(X_test)
        # #print(df_tweets.head())
        # print(df_tweets.Label.value_counts())
        print(get_n_imp_features(pipeline.named_steps["classifier"], pipeline.named_steps["vectorizer"].get_feature_names()))
        imps_a += list(get_n_imp_features(pipeline.named_steps["classifier"], pipeline.named_steps["vectorizer"].get_feature_names())[0])
        #imps_b += list(get_n_imp_features(pipeline.named_steps["classifier"], pipeline.named_steps["vectorizer"].get_feature_names())[1])
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        Precision.append(metrics.precision_score(y_test, predicted, average="weighted"))
        Recall.append(metrics.recall_score(y_test, predicted, average="weighted"))
        f1_scores.append(metrics.f1_score(y_test, predicted, average="weighted"))
        # for idx, pred in enumerate(predicted):
        #     print("predicted {} actual {}".format(pred, y_test.values[idx]))
        # print("================="*5)
        # print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted))
        # print("Logistic Regression Precision:", metrics.precision_score)
        # print("Logistic Regression Recall:", metrics.recall_score(y_test, predicted, average="weighted"))

print("avg accuracy", np.mean(accuracy))
print("avg Precision", np.mean(Precision))
print("avg recall", np.mean( Recall))
print("avg f1_Score", np.mean( f1_scores))
print("imps_a", set(imps_a), len(imps_a), len(set(imps_a)))
print("imps_b", set(imps_b), len(imps_b), len(set(imps_b)))



