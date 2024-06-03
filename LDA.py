import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pyLDAvis.lda_model
import matplotlib.pyplot as plt

def GetTopics(model, vectorizer, wordsPerTopic):
    words = vectorizer.get_feature_names_out()
    topicDFs = []
    for idx, topic in enumerate(model.components_):
        #sorts words on probability of being in topic
        topic_words = [words[i] for i in topic.argsort()[:-wordsPerTopic - 1:-1]]
        topic_word_frequencies = [topic[i] for i in topic.argsort()[:-wordsPerTopic - 1:-1]]
        
        topic_words = topic_words[::-1]
        topic_word_frequencies = topic_word_frequencies[::-1]
        
        df = pd.DataFrame({'Word': topic_words, 'Frequency': topic_word_frequencies})
        topicDFs.append(df)             
    return topicDFs
        