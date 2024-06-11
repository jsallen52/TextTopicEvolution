import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import plotly.graph_objects as go

# textColumnName = 'cm_probableCause'
# dateColumnName = 'cm_eventDate'

# df = pd.read_json('2020_2023.json')

# df = df[[dateColumnName, textColumnName]]
# df = df.dropna(subset=[textColumnName, dateColumnName], inplace=False) 
# df[dateColumnName] = pd.to_datetime(df[dateColumnName]).dt.tz_localize(None)

# df['Month'] = df[dateColumnName].dt.to_period('M')

# df = df.groupby('Month')[textColumnName].apply(lambda x: ' '.join(x)).reset_index()

# vectorizer = TfidfVectorizer(stop_words='english',  max_df=0.95)
# docTermMatrix = vectorizer.fit_transform(df[textColumnName])

# lastRow = pd.DataFrame(docTermMatrix.toarray(), columns=vectorizer.get_feature_names_out()).tail(1).reset_index(drop=True).T

# lastRowSorted = lastRow.sort_values(0, ascending=False)


# topTenArray = lastRowSorted.index[:10].values

# print(topTenArray)

def GetTopWords(df, textColumnName, dateColumnName):
    df['Month'] = df[dateColumnName].dt.to_period('M')

    interval_df = df.groupby('Month')[textColumnName].apply(lambda x: ' '.join(x)).reset_index()
    
    vectorizer = TfidfVectorizer(stop_words='english',  max_df=0.95, min_df=2)
    docTermMatrix = vectorizer.fit_transform(interval_df[textColumnName])

    lastRow = pd.DataFrame(docTermMatrix.toarray(), columns=vectorizer.get_feature_names_out()).tail(1).reset_index(drop=True).T

    lastRowSorted = lastRow.sort_values(0, ascending=False)

    topTenArray = lastRowSorted.index[:10].values
    
    return topTenArray



def CreateWordsOverTimeChart(df, textColumnName, DateColumnName, docTermMatrix, featureNames):
    # vectorizer = CountVectorizer(stop_words='english',  max_df=0.95)
    # docTermMatrix = vectorizer.fit_transform(df[textColumnName])
    # featureNames = vectorizer.get_feature_names_out()
    
    word_list = GetTopWords(df, textColumnName, DateColumnName)
    
    # Convert the result to a DataFrame
    word_counts_df = pd.DataFrame(docTermMatrix.toarray(), columns=featureNames)
    word_counts_df['date'] = df['TimeInterval']

    # Group by date and sum the counts for each date
    word_counts_by_date = word_counts_df.groupby('date').sum().reset_index()
    word_counts_by_date['date'] = word_counts_by_date['date'].dt.to_timestamp()

    # Create a Plotly figure
    fig = go.Figure()

    # Add a trace for each word
    for word in word_list:
        fig.add_trace(go.Scatter(x=word_counts_by_date['date'], y=word_counts_by_date[word], mode='lines', name=word))

    # Update layout
    fig.update_layout(
        title='Word Counts Over Time',
        xaxis_title='Date',
        yaxis_title='Word Count',
        legend_title='Words',
    )

    # Show the plot
    return fig