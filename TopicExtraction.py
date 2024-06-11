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

def GetTopicDocumentStats(dfTopicDistributions, numTopics, topicColors):
    dfStats = pd.DataFrame(columns=['Average', 'Standard Deviation'])
    topicData = []
    for i in range(numTopics):
        topicDocs = dfTopicDistributions[dfTopicDistributions['Topic'] == i].iloc[:,i]
        avg = topicDocs.mean()
        stdDev = topicDocs.std()
        dfStats.loc[i, 'Average'] = avg
        dfStats.loc[i, 'Standard Deviation'] = stdDev
        topicData.append(topicDocs)
        
    # Make Box Plot
    fig = go.Figure()
    for i, data in enumerate(topicData):
        fig.add_trace(go.Box(y=data, name=f'Topic {i+1}', marker_color=topicColors[i]))
    
    # Return the figure
    return fig
    
    
