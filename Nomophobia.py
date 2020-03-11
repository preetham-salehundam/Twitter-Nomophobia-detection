import gensim
import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score
from preprocessing import Tweet_preprocessor,Standardize

import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
import preprocessor as p

from sklearn.decomposition import NMF, LatentDirichletAllocation

import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaModel, LdaMulticore,CoherenceModel

import matplotlib.pyplot as plt

def IRR_calculator():
    kappa = cohen_kappa_score(data_tweets['W_Label'], data_tweets['Label'], labels=['Relevant', "Irrelevant"])

    data_tweets_Agreement = data_tweets[data_tweets["W_Label"] == data_tweets["Label"]]

    data_tweets['Agreement'] = np.where(data_tweets["W_Label"] == data_tweets["Label"], 1, 0)

    total_relevant = data_tweets.Agreement.value_counts()[1]
    total_annotated = data_tweets.Agreement.count()

    accuracy = (total_relevant / total_annotated) * 100

    print("Percentage accuracy", accuracy)
    print("cohen_kappa", kappa)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

        # print(topic.argsort())
        # print(topic)
        # print(topic.argsort()[:-no_top_words-1])
        # print(topic.argsort()[:-no_top_words - 1:-1])

def tweets_tokens(tweets):
    # split the tweets into tokens
    tokens = [[token for token in tweet.split()] for tweet in tweets]

    return tokens

#creating bigrams
def creating_bigram(tokens):
    # Build the bigram models
    phrase = gensim.models.phrases.Phrases(tokens, min_count=1, threshold=10)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram = gensim.models.phrases.Phraser(phrase)
    #print([bigram[tweet] for tweet in tokens])
    return [bigram[tweet] for tweet in tokens]

def dictionary_genism(tokens):

    # Create Dictionary
    dictionary = corpora.Dictionary(tokens)

    print(dictionary,dictionary.token2id)

    return dictionary

def corpus_genism(tweets,dictionary):
    tokenized_list = [simple_preprocess(tweet) for tweet in tweets]
    print(tokenized_list)
    corpus = [dictionary.doc2bow(token, allow_update=True) for token in tokenized_list]
    return corpus,tokenized_list

#eta=[0.01]*len(dictionary.keys()),alpha=[0.001]*num_top chunksize=100, alpha is documnet topic den
def compute_coherence_values(corpus, tweets_list,dictionary, num_top):
    lda_model = LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_top,
                                           random_state=100,
                                           passes=100,
                                           alpha=0.001,
                                           eta='auto',
                                           per_word_topics=True)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=tweets_list, dictionary=dictionary, coherence='c_v')

    return coherence_model_lda.get_coherence(),lda_model

# from wordcloud impor WordCloud
def word_cloud(processed_tweets):
    from wordcloud import WordCloud

def topic_visuvalization(model,corpus,dict):
    import pyLDAvis.gensim
    import pyLDAvis
    # Visualize the topics
    #pyLDAvis.enable_notebook()

    lda_data = pyLDAvis.gensim.prepare(model, corpus, dict, mds='mmds')
    pyLDAvis.show(lda_data)




if __name__ == "__main__":

    #data_tweets=pd.read_excel("/Users/vaishnaviv/PycharmProjects/Ml-Final/Jan31st_set1_labelledTweet.xls")
    data_tweets = pd.read_excel("/Users/vaishnaviv/PycharmProjects/Tweets data/Labelled Tweets Data/IRR/Nomophobia_reannotated_tweets.xls")
    #data_tweets = data_tweets.drop_duplicates(inplace=True,keep='first')
    data_tweets=data_tweets.drop_duplicates(subset=None, keep='first', inplace=False)
    data_tweets=data_tweets[:1582]
    print(data_tweets.Label.unique())
    print(data_tweets.W_Label.unique())
    #print(data_tweets.shape)
    print(data_tweets.shape)
    #Calculating IRR
    #IRR_calculator()

    data_tweets=data_tweets[data_tweets["Label"] == 'Relevant']
    #print(data_tweets['Label'])
    X=data_tweets['Text']
    y=data_tweets['Label']
    # #ngram = (1, 2)
    # bow_vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
    # # print(bow_vectorizer)
    # tfidf_vectorizer = TfidfVectorizer(min_df=0.05,max_df=0.95)

    pipeline = Pipeline(steps=[("preprocessor", Tweet_preprocessor(preprocessor=p))])
    tweets_list=pipeline.fit_transform(X)
    print(X[:2])

    #creating tokens sequence for each tweet
    tokens=tweets_tokens(tweets_list)
    #creating bi-grams for sequence of tokens
    bi_grams=creating_bigram(tokens)
    # Creating  Dictionary object that maps each word to a unique id.
    dictionary=dictionary_genism(bi_grams)
    print(dictionary)

    '''creating corpus a ‘corpus’ is typically a ‘collection of documents as a bag of words’. 
        That is, for each document, a corpus contains each word’s id and its frequency count in that document.
         As a result, information of the order of words is lost '''

    # Create Corpus
    corpus,tokenized_list=corpus_genism(tweets_list,dictionary)
    #view original text
    word_counts = [[(dictionary[id], count) for id, count in line] for line in corpus]
    print(word_counts)
    topics_count=[]
    coherance_values=[]
    perplexity=[]

    #model= None

    for num_top in range(9,10,1):
        topics_count.append(num_top)
        c_v,model=compute_coherence_values(corpus,tokenized_list,dictionary,num_top)
        coherance_values.append(c_v)
        perplexity.append(model.log_perplexity(corpus))
        for topics in model.print_topics():
            print(topics)
        #print(model[corpus])
        topic_visuvalization(model,corpus,dictionary)

    #plot for coherence score and number of topics
    print(coherance_values)
    plt.plot(topics_count,coherance_values,'bo-')
    plt.title("Optimal number of topics")
    plt.xlabel("Number of topics")
    plt.ylabel(" coherance score")
    plt.show()

    #plot for perplexity(Perplexity is a statistical measure of how well a probability model predicts a sample) lower the score better the measure and number of topics
    plt.plot(topics_count, perplexity, 'bo-')
    plt.title("Optimal number of topics")
    plt.xlabel("Number of topics")
    plt.ylabel(" perplexity score")
    plt.show()


    # feature_names=pipeline.named_steps["vectorizer"].get_feature_names()
    # #print(feature_names[380],feature_names[142])
    # #print(feature_names)
    #
    #
    #
    # # Run LDA
    # lda = LatentDirichletAllocation(n_topics=2, max_iter=5, learning_method='online', learning_offset=50.,
    #                                 random_state=0).fit(term_frequency)
    #
    # #print(lda)
    # display_topics(lda,feature_names,20)
    # #print(lda.components_)
    # print("perpexity",lda.perplexity(term_frequency))
    # print("score log likelihood",lda.score(term_frequency))



