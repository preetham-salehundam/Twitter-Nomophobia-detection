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
from sklearn.decomposition import SparsePCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
import preprocessor as p
from word_replacement_corpus import CONTRACTION_MAP, CUSTOM_STOP_WORDS, SLANG_WORDS
import re
import demoji
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.activations import sigmoid
import wordninja
#moved to setup.py
#demoji.download_codes()
# gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors, fasttext
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
# from sklearn.externals.six import StringIO
# from sklearn.tree import export_graphviz
# import pydotplus
# from IPython.display import Image
#using smote to deal imbalance
#from imblearn.over_sampling import SMOTE
# this symbol seems to have higher weightage in the final words when Naive Bayes is used,
# so adding it to punctuations to filter
punctuations = string.punctuation+"".join(["...", ".........." , "....","-", "--","/" , "&amp", "&lt", "&gt"])
nlp = spacy.load("en_core_web_sm")

STOP_WORDS = STOP_WORDS.union(CUSTOM_STOP_WORDS)

# excluding NO from stopwords for our use
STOP_WORDS.discard("no")
STOP_WORDS.discard("not")
STOP_WORDS.discard("off")



parser = English()
p.set_options(p.OPT.EMOJI, p.OPT.URL, p.OPT.SMILEY, p.OPT.NUMBER, p.OPT.MENTION)
#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)
single_char_pattern = re.compile("\s\w{1}\s", flags=re.MULTILINE)
single_digit_pattern = re.compile("\d+", flags=re.MULTILINE)
emoticons_happy = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D',
                   '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P',
                   ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'}
emoticons_sad = {':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/',
                 '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}
#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)

def clean_tweets(tweet):
    stop_words = STOP_WORDS
    # after tweepy preprocessing the colon symbol left remain after      #removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    # remove single chars
    tweet = single_char_pattern.sub(r' ', tweet)
    tweet = single_digit_pattern.sub(r' ', tweet)
    tweet = demoji.replace(tweet)
    # filter using NLTK library append it to a string
    word_tokens = tweet.split()
    word_tokens = [SLANG_WORDS.get(CONTRACTION_MAP.get(w,w), CONTRACTION_MAP.get(w,w)) for w in word_tokens]
    filtered_tweet = []
    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in punctuations:
            # return uncontracted word if present in map else return the same word as default
            #filtered_tweet.append(CONTRACTION_MAP.get(w, w))
            filtered_tweet.append(w.lower())
    return ' '.join(filtered_tweet)
    # print(word_tokens)hgdsl,cv
    # print(filtered_sentence)return tweet

class Tweet_preprocessor(TransformerMixin):
    def __init__(self, preprocessor):
        self.p = preprocessor

    def transform(self, X, **transform_params):
        df = pd.DataFrame(X)["Text"].apply(p.clean).apply(clean_tweets)
        return df.tolist()

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self):
        return {}

def tokenizer(sentence, lemmatize=True, remove_stops=True):
    """
    :param sentence:
    :param lemmatize:
    :param remove_stops:
    :return:
    """
    my_tokens = nlp(sentence)
    #lemmetization
    if lemmatize:
        my_tokens = [word.lemma_.lower().strip() for word in my_tokens if word.lemma_ !="-PRON-"]
    #remove stop words
    if remove_stops:
        my_tokens = [word for word in my_tokens if word not in STOP_WORDS and word not in punctuations]
    return my_tokens

# def filter_tokens(tokens, filter):
#     return [token for token in tokens if token not in filter]

#unused
class Csr2Dense(TransformerMixin):
    def __init__(self):
        pass;

    def transform(self, X, **transform_params):
        return X.toarray()

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self):
        return {}

class Csr2PCA(TransformerMixin):
    def __init__(self, pca):
        self.pca = pca

    def transform(self, X, **transform_params):
        #X = X.toarray()
        return self.pca.fit_transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self):
        return {}

class Word2vec(TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, **transform_params):
        #X = X.toarray()
        df = pd.DataFrame({"Text":X})["Text"].apply(lambda x: " ".join(x)).apply(nlp)
        vec = df.apply(lambda x: x.vector)
        return vec.tolist()

    def fit(self, X, y=None, **fit_params):
        #self.resampled = self.smote.fit_resample(X,y)
        return self

    def get_params(self):
        return {}


class DNN(TransformerMixin):
    def __init__(self ,n_class = 2 ):
        self.model = Sequential()
        #self.n_layers= n_layers
        self.n_class = n_class
        self.output = None
    def transform(self, X, **transform_params):
        return self.output

    def fit(self, X, y=None, **fit_params):
        X = np.array(X)
        self.model.add(Dense(units=8, input_dim=X.shape[-1] , use_bias=True, activation="relu", kernel_regularizer="l2"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(units=4, activation="relu", use_bias=True, kernel_regularizer=l2(l=0.001)))
        self.model.add(Dense(units=self.n_class, activation="sigmoid"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())
        self.output = self.model.fit(X, to_categorical(y), epochs=20, batch_size=8)
        return self

    def predict(self,X):
        df=pd.DataFrame({"X":X}, index=1).apply(self._predict, axis=1)
        return df.apply(np.argmax)
    def _predict(self, X):
        return self.model.predict(X[0].reshape(1,-1))
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
        pos_dict = {k:v for k,v in zip(pos, np.take(classifier.feature_log_prob_[0, :], pos_class_prob_sorted[:n]))}
        neg_dict = {k:v for k,v in zip(neg, np.take(classifier.feature_log_prob_[1, :], neg_class_prob_sorted[:n]))}
        return pos_dict, neg_dict
    if isinstance(classifier, LogisticRegression):
        feature_imps = classifier.coef_[0].argsort()[::-1]
        imp_features = np.take(feature_names, feature_imps[:n])
        imp_dict = {k:v for k,v in zip(imp_features, np.take(classifier.coef_[0], feature_imps[:n]))}
        return imp_dict, None
    if isinstance(classifier, DecisionTreeClassifier):
        feature_imps = classifier.feature_importances_.argsort()[::-1]
        imp_features = np.take(feature_names, feature_imps[:n])
        imp_dict = {k: v for k, v in zip(imp_features, np.take(classifier.feature_importances_, feature_imps[:n]))}
        return  imp_dict, None
    if isinstance(classifier, RandomForestClassifier):
        feature_imps = classifier.feature_importances_.argsort()[::-1]
        imp_features = np.take(feature_names, feature_imps[:n])
        imp_dict = {k: v for k, v in zip(imp_features, np.take(classifier.feature_importances_, feature_imps[:n]))}
        return imp_dict, None
    if isinstance(classifier, LinearSVC):
        feature_imps = classifier.coef_[0].argsort()[::-1]
        imp_features = np.take(feature_names, feature_imps[:n])
        imp_dict = {k: v for k, v in zip(imp_features, np.take(classifier.coef_[0], feature_imps[:n]))}
        return imp_dict, None


if __name__ == "__main__":
    #/Users/vaishnaviv/PycharmProjects/Ml-Final/project_8_labels_Wed_Nov_13_2019.csv
    df_tweets = pd.read_csv("project_8_labels_Wed_Nov_24_2019.csv")

    df_tweets["Label"][(df_tweets["Label"] < 5)] = 1
    df_tweets["Label"][(df_tweets["Label"] >= 5)] = 0
    # N = len(df_tweets[df_tweets["Label"] == 1])
    # df_tweets_0 = df_tweets[df_tweets["Label"] == 0].sample(n=N, random_state=42)
    # df_tweets_0.head(10).to_csv("neg.csv")
    # df_tweets_1 = df_tweets[df_tweets["Label"] == 1].sample(n=N, random_state=42)
    # df_tweets_1.head(10).to_csv("pos.csv")
    # df_tweets = df_tweets_0.append(df_tweets_1, ignore_index = True)
    #df_tweets["Text"].apply(tokenizer)
    bow_vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1,3))
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,3))
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

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    accuracy=[]
    Precision=[]
    Recall=[]
    f1_scores=[]
    imps=[]
    imps_a = []
    imps_b = []

    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
    classifier_lr = LogisticRegression(class_weight='balanced')
    classifier_NB = MultinomialNB(fit_prior=True)
    classifier_tree = DecisionTreeClassifier(criterion="gini")
    classifier_RF = RandomForestClassifier()
    classifier_svc = LinearSVC()

    #glove_file = datapath('glove.6B/glove.6B.100d.txt')
    # tmp_file = get_tmpfile("word2vec_100d.txt")
    # #_ = glove2word2vec(glove_file, tmp_file)
    # model = KeyedVectors.load_word2vec_format(tmp_file)
    # print(model["king"])

    # model = Sequential()
    # model.add(Dense(units=300, input_shape=X.shape[-1]))
    # model.add(Dense(units=256, activation="sigmoid"))
    # model.add(Dense(units=128, activation="sigmoid"))
    # model.add(Dense(units=64, activation="sigmoid"))
    # model.add(Dense(units=2, activation="sigmoid"))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    i = 1
    for train, test in cv.split(X,y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_test, return_counts=True))

        #pipeline = Pipeline(steps=[("vectorizer", word2vec(datafile="word2vec_100d.txt", vectorizer=tokenizer)),("classifier",classifier_tree)])
        pipeline = Pipeline(steps=[("preprocessor", Tweet_preprocessor(preprocessor=p)), ("vectorizer", tfidf_vectorizer), ("converter", Csr2Dense()), ("classifier", DNN())]) #("dimen2sionality_reducer", Csr2PCA(SparsePCA(n_components=10)))
        #pipeline = Pipeline(steps=[("preprocessor", Tweet_preprocessor(preprocessor=p)), ("vectorizer", tfidf_vectorizer), ("classifier", classifier_NB)]) #("dimensionality_reducer", Csr2PCA(SparsePCA(n_components=10)))
        pipeline.fit(X_train, y_train)
        predicted=pipeline.predict(X_test)
        # #print(df_tweets.head())
        # print(df_tweets.Label.value_counts())
        # dot_data = StringIO()
        # if isinstance(pipeline.named_steps["classifier"], DecisionTreeClassifier):
        #     export_graphviz(pipeline.named_steps["classifier"], out_file=dot_data,
        #                     filled=True, rounded=True,
        #                     special_characters=True)
        #     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        #     with open("image.png", "wb") as fd:
        #         fd.write(graph.create_png())




        # print(get_n_imp_features(pipeline.named_steps["classifier"], pipeline.named_steps["vectorizer"].get_feature_names(), n=20))
        # imps_a += list(get_n_imp_features(pipeline.named_steps["classifier"], pipeline.named_steps["vectorizer"].get_feature_names(), n=20)[0])
        # imps_b += list(get_n_imp_features(pipeline.named_steps["classifier"], pipeline.named_steps["vectorizer"].get_feature_names(), n=20)[1]) if isinstance(pipeline.named_steps["classifier"], MultinomialNB) else []
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        Precision.append(metrics.precision_score(y_test, predicted, average="weighted"))
        Recall.append(metrics.recall_score(y_test, predicted, average="weighted"))
        f1_scores.append(metrics.f1_score(y_test, predicted, average="weighted"))
        print(metrics.confusion_matrix(y_test, predicted))
        df = pd.DataFrame({"X":X_test,"y": y_test})
        df[df["y"] == 1].to_csv("pos_pred_{}.csv".format(i), index=False)
        df[df["y"] == 0].to_csv("neg_pred_{}.csv".format(i), index=False)
        # for idx, pred in enumerate(predicted):
        #     print("predicted {} actual {}".format(pred, y_test.values[idx]))
        # print("================="*5)
        # print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted))
        # print("Logistic Regression Precision:", metrics.precision_score)
        # print("Logistic Regression Recall:", metrics.recall_score(y_test, predicted, average="weighted"))
        i+=1
print("avg accuracy", np.mean(accuracy))
print("avg Precision", np.mean(Precision))
print("avg recall", np.mean( Recall))
print("avg f1_Score", np.mean( f1_scores))
print("imps_a", sorted(set(imps_a)), len(imps_a), len(set(imps_a)))
print("imps_b", sorted(set(imps_b)), len(imps_b), len(set(imps_b)))
print("unique_a", set(imps_a).difference(set(imps_b)))
print("unique_b", set(imps_b).difference(set(imps_a)))



