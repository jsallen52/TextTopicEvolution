import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import plotly.graph_objects as go

def GetTopRecentWordsTFIDF(df, textColumnName, dateColumnName, numWords):
    df['Month'] = df[dateColumnName].dt.to_period('M')

    interval_df = df.groupby('Month')[textColumnName].apply(lambda x: ' '.join(x)).reset_index()
    
    vectorizer = TfidfVectorizer(stop_words='english',  max_df=0.95)
    docTermMatrix = vectorizer.fit_transform(interval_df[textColumnName])

    lastRow = pd.DataFrame(docTermMatrix.toarray(), columns=vectorizer.get_feature_names_out()).tail(1).reset_index(drop=True).T

    lastRowSorted = lastRow.sort_values(0, ascending=False)

    topTenArray = lastRowSorted.index[:numWords].values
    
    return topTenArray

def GetFlaggedRecentWordsZscore(df, textColumnName, dateColumnName, numWords):
    df['Month'] = df[dateColumnName].dt.to_period('M')

    vectorizer = CountVectorizer(stop_words='english')
    docTermMatrix = vectorizer.fit_transform(df[textColumnName])

    terms = vectorizer.get_feature_names_out()
    docTermDF = pd.DataFrame(docTermMatrix.toarray(), columns=terms)
    # docTermDF[docTermDF > 1] = 1

    docTermDF['Month'] = df['Month'].values

    docTermDF = docTermDF.groupby('Month').sum()

    # Compute the mean and standard deviation for each term
    mean = docTermDF.mean(axis=0)
    # ddof=0 for population standard deviation
    # ddof=1 for sample standard deviation
    std = docTermDF.std(axis=0, ddof=0)  

    n = len(docTermDF)
    z_scores = abs((docTermDF - mean) / std)  # t = (docTermDF - mean) / (std / (n ** 0.5))

    lastRow = z_scores.tail(1).reset_index(drop=True).T
    lastRowSorted = lastRow.sort_values(0, ascending=False)
    topTenArray = lastRowSorted.index[:numWords].values
    
    return topTenArray



def GetTopRecentWordsCTFIDF(df: pd.DataFrame, textColumnName: str, dateColumnName: str, numWords: int) -> np.ndarray:
    """
    Gets the words for hte most recent time interval using the C-TF-IDF method

    Args:
        df (pd.DataFrame): The input dataframe.
        textColumnName (str): The name of the text column.
        dateColumnName (str): The name of the date column.
        numWords (int): The number of top words to retrieve.

    Returns:
        np.ndarray: An array of top recent words.
    """
    df['Month'] = df[dateColumnName].dt.to_period('M')

    interval_df = df.groupby('Month')[textColumnName].apply(lambda x: ' '.join(x)).reset_index()
    
    vectorizer = CountVectorizer(stop_words='english')
    docTermMatrix = vectorizer.fit_transform(interval_df[textColumnName])
    
    totalFrequencies = docTermMatrix.sum(axis=0)
    avgTotalWords = docTermMatrix.sum() / docTermMatrix.shape[0]
    adjustedIDF = np.log(1 + avgTotalWords/totalFrequencies)
    docTermMatrix = docTermMatrix.multiply(adjustedIDF)

    lastRow = pd.DataFrame(docTermMatrix.toarray(), columns=vectorizer.get_feature_names_out()).tail(1).reset_index(drop=True).T

    lastRowSorted = lastRow.sort_values(0, ascending=False)

    topTenArray = lastRowSorted.index[:numWords].values
    
    return topTenArray

def CreateWordsOverTimeChart(df: pd.DataFrame, textColumnName: str, dateColumnName: str) -> go.Figure:
    """
    Creates a line graph using Plotly with word frequencies over time for the flagged words found in the most recent time interval

    Args:
        df (pd.DataFrame): The input dataframe.
        textColumnName (str): The name of the text column.
        dateColumnName (str): The name of the date column.

    Returns:
        go.Figure: The Plotly figure object for the line graph, or None if there is no data to plot.
    """
    vectorizer = CountVectorizer(stop_words='english')
    docTermMatrix = vectorizer.fit_transform(df[textColumnName])
    featureNames = vectorizer.get_feature_names_out()
    
    word_list = GetTopRecentWordsCTFIDF(df, textColumnName, dateColumnName, 10)
    
    word_counts = pd.DataFrame(docTermMatrix.toarray(), columns=featureNames)
    word_counts['TimeInterval'] = df['TimeInterval'].values
    
    # Marks which documents have each word rather than the count of each word for each document
    for word in word_list:
        word_counts.loc[word_counts[word] > 0, word] = 1
        word_counts.loc[word_counts[word] == 0, word] = 0


    # Group by timestamps and sum the word counts
    df_grouped = word_counts.groupby('TimeInterval')[word_list].sum().reset_index()
    
    df_grouped['TimeInterval'] = df_grouped['TimeInterval'].dt.to_timestamp()

    #minimum number of documents for word to be included in chart
    minDocumentCount = 0
 
    # Create line graph with plotly
    data = []
    for word in word_list:
        if df_grouped[word].tail(1).values[0] >= minDocumentCount:
            data.append(go.Scatter(x=df_grouped['TimeInterval'], y=df_grouped[word], mode='lines+markers', name=word))

    layout = go.Layout(title='',
                    xaxis=dict(title='Time Interval'),
                    yaxis=dict(title='Document Count'),
                    showlegend=True,
                    margin=dict(t=20, b=0, l=0, r=0))  # Show the legend even if there is only one plot

    if(len(data) > 0):
        fig = go.Figure(data=data, layout=layout)
        # Show the plot
        return fig

    return None
