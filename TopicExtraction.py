import pandas as pd
import plotly.graph_objects as go

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

def GetTopicDocumentStats(dfTopicDistributions, numTopics, topicColors, topicProbColumn=None):
    dfStats = pd.DataFrame(columns=['Average', 'Standard Deviation'])
    topicData = []
    for i in range(numTopics):
        topicDocs = dfTopicDistributions[dfTopicDistributions['Topic'] == i]

        if(topicProbColumn is None):
            topicDocs = topicDocs.iloc[:,i]
        else:
            topicDocs = topicDocs[topicProbColumn]
            
        avg = topicDocs.mean()
        stdDev = topicDocs.std()
        dfStats.loc[i, 'Average'] = avg
        dfStats.loc[i, 'Standard Deviation'] = stdDev
        topicData.append(topicDocs)
        
    # Make Box Plot
    fig = go.Figure()
    fig.update_layout(
    title="",  # Remove the title
    margin=dict(t=20, b=0, l=0, r=0),  # Set all margins to 0
    )

    for i, data in enumerate(topicData):
        fig.add_trace(go.Box(y=data, name=f'Topic {i+1}', marker_color=topicColors[i]))

    return fig


def GetTopicsBERT(bertModel):
    """
    Given a BERTopic modelreturns a DataFrame
    containing the words and their corresponding frequencies for each topic.

    Args:
        bertModel: BERTopic fit model

    Returns:
        A list of pandas DataFrames containing the words and their corresponding frequencies
        for each topic.
    """

    # Initialize an empty list to store the DataFrames for each topic
    topicDFs = []
    
    topicInfo = bertModel.get_topics()

    # Iterate over each topic and create a DataFrame
    for topic_num, words_scores in topicInfo.items():
        if(topic_num == -1):
            continue
        
        # Create a DataFrame for the current topic
        df = pd.DataFrame(words_scores, columns=["Word", "Frequency"]).iloc[::-1]
        
        # Store the DataFrame in the dictionary
        topicDFs.append(df)
    
    return topicDFs
    
