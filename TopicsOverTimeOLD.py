import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from PrepareData import GetDataFrame

def PlotOverTime(ax, grouped_df, X, vectorizer, topic_words, lineColor): 
    word_frequencies_sum = np.zeros(X.shape[0])

    # Loop through the topic words and add their counts to the sum
    for word in topic_words:
        if word in vectorizer.vocabulary_:
            word_index = vectorizer.vocabulary_[word]
            word_frequencies_sum += X.toarray()[:, word_index]

    sum_columns = word_frequencies_sum

    grouped_df['Topic_Frequency'] = sum_columns

    grouped_df.to_csv('grouped_df.csv', index=False)
    ax.plot(grouped_df.index, grouped_df['Topic_Frequency'], color=lineColor)
    
def CreateTimeFig(df, primary_color, topicDFs, topicColors, textColumnName):
    # group data frame by month and combine the text column
    grouped_df = df.groupby('Month')[textColumnName].agg(lambda x: ' '.join(x)).reset_index() 

    vectorizer = TfidfVectorizer(stop_words='english')
    #vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(grouped_df[textColumnName])
    
    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
    
    i = 0
    for df in topicDFs:
        PlotOverTime(ax, grouped_df, X, vectorizer, df['Word'].tolist(), topicColors[i])
        i += 1
        

    ax.set_xlabel('Month', color=primary_color)
    ax.set_ylabel('Topic Frequency', color=primary_color)
    ax.set_ylim(bottom=0)
    # ax.set_ylim(top=1)
    ax.set_title('Topic Frequency Change Over Time', color=primary_color)

    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('none')

    ax.spines['bottom'].set_color(primary_color)  
    ax.spines['left'].set_color(primary_color) 
    ax.tick_params(axis='x', colors=primary_color)
    ax.tick_params(axis='y', colors=primary_color)

    # Show horizontal grid lines
    ax.grid(axis='y', linestyle='-', linewidth='0.5', color='grey')

    # Month names
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Setting the x-axis labels to month names
    ax.set_xticks(ticks=range(12), labels=months, rotation=45)
    
    return fig

