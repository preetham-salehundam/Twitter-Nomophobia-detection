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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics
import preprocessor as p
import re
import demoji
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
from imblearn.over_sampling import SMOTE
punctuations = string.punctuation

# stop words
nlp = spacy.load("en")
stop_words = STOP_WORDS
# excluding NO from stopwords for our use
STOP_WORDS.discard("no")

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

    tweet = demoji.replace(tweet)
    # filter using NLTK library append it to a string
    word_tokens = tweet.split()
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
    # print(word_tokens)
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



class word2vec(TransformerMixin):
    def __init__(self, datafile, vectorizer):
        self.model = KeyedVectors.load_word2vec_format(fname=datafile)
        self.vectorizer = vectorizer
        self.vec2word = {}

    def transform(self, X, **transform_params):
        df = pd.DataFrame(X)["Text"].apply(self.vectorizer)
        vec = df.apply(self.get_vectors)
        return vec

    def fit(self, X, y=None, **fit_params):
        #self.resampled = self.smote.fit_resample(X,y)
        return self

    def get_params(self):
        return {}

    def get_vectors(self, tokens):
        agg_vec = {}
        vec = np.zeros((100,))
        for token in tokens:
            try:
                vec += np.array(self.model[token])
            except Exception as e:
                vec += np.zeros((100,))
                print(e)
        #agg_vec[index] = vec/len(tokens)
        return vec/len(tokens)



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
    if isinstance(classifier, RandomForestClassifier):
        feature_imps = classifier.feature_importances_.argsort()[::-1]
        imp_features = np.take(feature_names, feature_imps[:n])
        return imp_features, None



if __name__ == "__main__":

    df_tweets = pd.read_csv("project_8_labels_Wed_Nov_24_2019.csv")

    df_tweets["Label"][(df_tweets["Label"] < 5)] = 1
    df_tweets["Label"][(df_tweets["Label"] >= 5)] = 0
    N = len(df_tweets[df_tweets["Label"] == 1])
    df_tweets_0 = df_tweets[df_tweets["Label"] == 0].sample(n=N)
    df_tweets_0.head(10).to_csv("neg.csv")
    df_tweets_1 = df_tweets[df_tweets["Label"] == 1].sample(n=N)
    df_tweets_1.head(10).to_csv("pos.csv")
    df_tweets = df_tweets_0.append(df_tweets_1, ignore_index = True)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    accuracy=[]
    Precision=[]
    Recall=[]
    f1_scores=[]
    imps=[]
    imps_a = []
    imps_b = []

    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    classifier = LogisticRegression(class_weight='balanced')
    classifier_NB = MultinomialNB(fit_prior=True)
    classifier_tree = DecisionTreeClassifier(criterion="gini")
    classifier_RF = RandomForestClassifier()

    #glove_file = datapath('glove.6B/glove.6B.100d.txt')
    # tmp_file = get_tmpfile("word2vec_100d.txt")
    # #_ = glove2word2vec(glove_file, tmp_file)
    # model = KeyedVectors.load_word2vec_format(tmp_file)
    # print(model["king"])


    for train, test in cv.split(X,y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_test, return_counts=True))





        #pipeline = Pipeline(steps=[("vectorizer", word2vec(datafile="word2vec_100d.txt", vectorizer=tokenizer)),("classifier",classifier_tree)])
        pipeline = Pipeline(steps=[("preprocessor", Tweet_preprocessor(preprocessor=p)),("vectorizer", tfidf_vectorizer),("classifier",classifier_RF)])


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
        print(get_n_imp_features(pipeline.named_steps["classifier"], pipeline.named_steps["vectorizer"].get_feature_names()))
        imps_a += list(get_n_imp_features(pipeline.named_steps["classifier"], pipeline.named_steps["vectorizer"].get_feature_names())[0])
        imps_b += list(get_n_imp_features(pipeline.named_steps["classifier"], pipeline.named_steps["vectorizer"].get_feature_names())[1]) if isinstance(pipeline.named_steps["classifier"], MultinomialNB) else []
        accuracy.append(metrics.accuracy_score(y_test, predicted))
        Precision.append(metrics.precision_score(y_test, predicted, average="weighted"))
        Recall.append(metrics.recall_score(y_test, predicted, average="weighted"))
        f1_scores.append(metrics.f1_score(y_test, predicted, average="weighted"))
        print(metrics.confusion_matrix(y_test, predicted))
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



