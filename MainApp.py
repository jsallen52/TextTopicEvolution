import os
from sklearn.decomposition import NMF, PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from FlagFrequencies import CreateWordsOverTimeChart
from PrepareData import AssignTimeInterval, GetDataFrame, GetFullDataFrame
from DocumentsOverTime import CreateDocTimeFig
from TopicExtraction import GetTopicDocumentStats, GetTopics, GetTopicsBERT

import plotly.express as px
import plotly.graph_objects as go

from PartsOfSpeach import FilterForNouns

from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

additional_stop_words = []

# additional_stop_words= ['pilot', 'failure','control','resulted','loss','maintain','directional','flight','airplane','determined','reasons','available','based','contributing', 'improper', 'landing']

useAdditionalStopWords = True

dataFileName = '2020_2023.json'
#analysisNarrative
textColumnName = 'cm_probableCause'
dateColumnName = 'cm_eventDate'

# Title of the app
st.title('Text Topic Analysis')

# Find all .json or .csv files in the root folder
all_files = [f for f in os.listdir('.') if f.endswith(('.json', '.csv'))]

#------Cache Functions---------------------------------
# Streamlit will automatically cache the results of the 
# functions so any future calls with the same parameters will not be re-run
#------------------------------------------------------

@st.cache_resource
def loadBertModel(_documents, numTopics, reduceTopics, wordsPerTopic, minClusterSize, _vectorizer, textColumnName, dateColumnName, startDate, endDate, minDF, maxDF, minNGram,maxNGram):
    hdbscan = HDBSCAN(
        min_cluster_size=minClusterSize, 
        metric='euclidean', 
        cluster_selection_method='eom', 
        prediction_data=True
    )
    
    umap = UMAP(
        n_neighbors=15, 
        n_components=5, 
        metric='cosine',
    )

    # Create a BERTopic model and fit it to your documents
    bertModel = BERTopic(
        vectorizer_model=_vectorizer, 
        hdbscan_model=hdbscan, 
        nr_topics= (numTopics + 1) if reduceTopics else None, # +1 for outlier
        top_n_words=wordsPerTopic,
    )
    topics, probs = bertModel.fit_transform(_documents)
    
    return bertModel, topics, probs

@st.cache_resource
def loadLDA_NMF(selectedAlgo, numTopics, _docTermMatrix, textColumnName, dateColumnName, startDate, endDate, minDF, maxDF, minNGram,maxNGram):
    if(selectedAlgo == 'LDA'): 
        topicExtractionModel = LatentDirichletAllocation(n_components=numTopics, max_iter=50, learning_method='online')
        
    elif(selectedAlgo == 'NMF'):
        #NMF used TFIDF for vectorization
        topicExtractionModel = NMF(n_components=numTopics)
        
    return topicExtractionModel, topicExtractionModel.fit_transform(_docTermMatrix)

#-------Filters Side Bar-----------------------------------

dataFileName = st.sidebar.selectbox('Data Source', all_files, index = all_files.index(dataFileName))

with st.sidebar.form(key='filter_form'):
    st.header('General Filters')
    
    df = GetFullDataFrame(dataFileName)
    #Gets all columns that are non integers and so will be converted into strings
    all_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    textIndex = 0
    if(dataFileName == '2020_2023.json'):
        textIndex = all_columns.index(textColumnName)
    textColumnName = st.selectbox('Text Column',all_columns, index=textIndex, key="text_column_name")
    
    dateIndex = 0
    #Gets all columns with values that can be converted to dates
    date_columns = [col for col in all_columns if pd.to_datetime(df[col], errors='coerce').notnull().all()]
    if(dataFileName == '2020_2023.json'):
        dateIndex = date_columns.index(dateColumnName)
    dateColumnName = st.selectbox('Date Column', date_columns,index=dateIndex, key="date_column_name")
    
    # Create init dataframe
    df = GetDataFrame(dataFileName, textColumnName, dateColumnName)
    
    start_date, end_date = st.date_input(
        "Select Date Range",
        value=(df[dateColumnName].min(), df[dateColumnName].max()),
        format="MM/DD/YYYY",
        key="date_range"
    )
    
    options = ['Weekly', 'Monthly', 'Yearly']
    selectedTimeInterval = st.selectbox('Time Interval', options, index = 1)
    
    topWordCount = st.slider('Top Words', 5, 50, 20)
    
    with st.expander("Advanced Options"):
    
        minDocFreq = st.number_input('Minimum Document Frequency', 0, 100, 2)
        maxDocFreq = st.number_input('Maximum Document Frequency', 0.0, 1.0, .95)
        
        minNgram = st.number_input('Minimum NGram', 1, 3, 1)
        maxNgram = st.number_input('Maximum NGram', 1, 4, 1)
        
        #-------Side Bar Parts of Speech
        st.subheader('Parts of Speech')
        # Filter for nouns only
        useNounsOnly = st.checkbox("Nouns Only", value=True, key="nouns_only_checkbox")
        
        #-------Side Bar Stop Words
        st.subheader('Stop Words')
        useStopWords = st.checkbox("Use Standard Stop Words", value=True, key="stop_words_checkbox")

        vectorizer = CountVectorizer(stop_words= 'english' if useStopWords else None, ngram_range=(minNgram, maxNgram))
        X = vectorizer.fit(df[textColumnName])

        useAdditionalStopWords = st.checkbox("Use Additional Stop Words", value=True, key="my_checkbox2", disabled=not useStopWords)

        additional_stop_words = st.multiselect('Select Additional Stop Words', list(vectorizer.get_feature_names_out()), default=additional_stop_words, disabled= (not useAdditionalStopWords) or (not useStopWords))

        if(not useStopWords):
            useAdditionalStopWords = False
            
        #-------Side Bar Topic Extraction
        st.subheader('Topic Extraction')
        algoOptions = ['LDA', 'NMF', 'BERTopic']
        selectedAlgo = st.selectbox('Algorithm', algoOptions, index=2)
        
        numTopicsDisabled = (selectedAlgo == 'BERTopic')
        numTopics = st.slider('Topic Count', 3, 40, 10, disabled = numTopicsDisabled)
        wordsPerTopic = st.slider('Words Per Topic', 8, 15, 10)

        st.subheader('BERTopic Options')
        reduceTopics = st.checkbox("Reduce Topics", value=True, key="reduceTopicsBERT", )
        
        minClusterSize = st.slider('Minimum Cluster Size', 5, 50, 15)
        
    #-------Submit button
    st.form_submit_button(label='Apply Filters')
    
#-------------------------------------------------------------
df = df[(df[dateColumnName] >= pd.to_datetime(start_date)) & (df[dateColumnName] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))]

if(useNounsOnly and selectedAlgo != 'BERTopic'):
    df = FilterForNouns(df, textColumnName) 

df = AssignTimeInterval(df, dateColumnName, selectedTimeInterval)

# Create columns
col1, col2, col3 = st.columns(3)

# Define box content and styles
box_style = """
    <div style="
        background-color: #f0f0f0; 
        border-radius: 15px; 
        padding: 20px; 
        text-align: center; 
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    ">
        <h3 style="color: #7F7F7F">{title}</h3>
        <p style="font-size: 24px; color: #393939;">{number}</p>
    </div>
"""

documentCount = format(df.shape[0], ',')

all_stop_words = list(CountVectorizer(stop_words='english').get_stop_words()) if useStopWords else None

if(useAdditionalStopWords):
    all_stop_words = all_stop_words + additional_stop_words

vectorizer = CountVectorizer(stop_words=all_stop_words, max_df=maxDocFreq, min_df=minDocFreq, ngram_range=(minNgram, maxNgram))

docTermMatrix = vectorizer.fit_transform(df[textColumnName])
total_words = format(docTermMatrix.sum(), ',')
unique_words = format(len(vectorizer.get_feature_names_out()), ',')

# Display boxes in each column
with col1:
    st.markdown(box_style.format(title='Documents', number=documentCount), unsafe_allow_html=True)

with col2:
    st.markdown(box_style.format(title='Total Words', number=total_words), unsafe_allow_html=True)

with col3:
    st.markdown(box_style.format(title='Unique Words', number=unique_words), unsafe_allow_html=True)
    
#---Word Count Chart---------------------------------\
    
st.write('\n')
# Get feature names (words) from the vectorizer
feature_names = vectorizer.get_feature_names_out()
# Get the word counts for each feature (word) across all text entries
word_counts = docTermMatrix.sum(axis=0).A1
# Create a DataFrame to store feature names and their corresponding word counts
word_count_df = pd.DataFrame({'Word': feature_names, 'Word Count': word_counts})
# Sort the DataFrame by word counts in descending order to get the most common words
most_common_words = word_count_df.sort_values(by='Word Count', ascending=False)

# Plot the top words with their counts
topWords = most_common_words.head(topWordCount)
topWords = topWords[::-1]

countFigHeight = 400 + (topWordCount * 10)
fig = go.Figure(data=[go.Bar(x=topWords['Word Count'], y=topWords['Word'], orientation='h')])
# Remove axis labels
fig.update_layout(
    xaxis_title='', 
    yaxis_title='', 
    height = countFigHeight,
    margin=dict(l=0, r=0, t=50, b=0),
    title = f"Frequency of Top {topWordCount} Words"
)
# Remove y-axis numbers
fig.update_xaxes(showgrid=True)
fig.update_yaxes(
    tickmode='linear', 
    tick0=0, 
    showgrid=False, 
    title_standoff=0,  # Distance between axis title and tick labels
    ticklabelposition="inside",  # Position tick labels inside to save space
)

st.plotly_chart(fig, use_container_width=True)

#------Flagged Words Chart--------------------------
flagWordsChart = CreateWordsOverTimeChart(df, textColumnName,dateColumnName, docTermMatrix, feature_names)

if flagWordsChart is not None:
    st.plotly_chart(flagWordsChart, use_container_width=True)
 
#--Topic Extraction---------------------------------------------
st.markdown("---",)
st.subheader(f"Topic Extraction:", help='Each chart displays the top n (number can be slected in sidebar options) most descriptive words for each topic. ')

if(selectedAlgo == 'BERTopic'):
    documents = df[textColumnName].values
    
    bertModel, docTopics, probs = loadBertModel(documents, numTopics, reduceTopics, wordsPerTopic, minClusterSize, vectorizer, textColumnName, dateColumnName, start_date, end_date, minDocFreq, maxDocFreq, minNgram,maxNgram)
    
    numTopics = len(bertModel.get_topic_info()) - 1
    
    topicDFs = GetTopicsBERT(bertModel)
    
    dfTopicDistributions = pd.DataFrame({'Probs': probs})
    dfTopicDistributions['Topic'] = docTopics
    
    representativeDocs = bertModel.get_representative_docs()

elif(selectedAlgo == 'LDA') or (selectedAlgo == 'NMF'):
    if(selectedAlgo == 'NMF'):
        vectorizer = TfidfVectorizer(stop_words=all_stop_words, max_df=maxDocFreq, min_df=minDocFreq, ngram_range=(minNgram, maxNgram))
        docTermMatrix = vectorizer.fit_transform(df[textColumnName])
        
    topicExtractionModel, documentTopicDistributions = loadLDA_NMF(selectedAlgo, numTopics, docTermMatrix, textColumnName, dateColumnName,start_date, end_date, minDocFreq, maxDocFreq, minNgram,maxNgram)

    topic_columns = [f'Topic {i}' for i in range(numTopics)]
    dfTopicDistributions = pd.DataFrame(documentTopicDistributions, columns=topic_columns)

    docTopics = np.argmax(documentTopicDistributions, axis=1)
    dfTopicDistributions['Topic'] = docTopics
    topicDFs = GetTopics(topicExtractionModel, vectorizer, wordsPerTopic)
    
df['Topic'] = docTopics

#---Topic Search Box-----------------------------
searchWord = st.text_input("Search Topics:")

if selectedAlgo =='BERTopic' and searchWord:
    similar_topics, similarity = bertModel.find_topics(searchWord, top_n=4)
    for i in range(4):
        if(similar_topics[i] >= 0):
            st.write(f"Topic {similar_topics[i] + 1}: {similarity[i]:.2f}")

#--Topic Word Charts-----------------------------

topicColors = [
    '#800000', 
    '#469990', 
    '#f58231', 
    '#3cb44b', 
    '#a9a9a9', 
    '#e6194B', 
    '#911eb4', 
    '#42d4f4', 
    '#aaffc3',
    '#9A6324',
    '#808000',
    '#000075',
    '#000000',
    '#ffe119',
    '#bfef45',
    '#4363d8',
    '#f032e6',
    '#fabed4',
    '#ffd8b1',
    '#fffac8',
    '#dcbeff',
]


topicColors = topicColors * 10

st.write('\n')

figHeights = 300
chartsPerRow = 4
for i in range(numTopics):
    if(i % chartsPerRow == 0):
        bar_cols= st.columns(chartsPerRow)
        
    with bar_cols[i % chartsPerRow]:
        fig = go.Figure(data=[go.Bar(x=topicDFs[i]["Frequency"], y=topicDFs[i]["Word"], orientation='h',  marker_color=topicColors[i])])
        # Remove axis labels
        fig.update_layout(xaxis_title='', yaxis_title='', height=figHeights)
        fig.update_layout(
        title={
            'text': f'Topic {i+1}',
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=0, r=0, t=30, b=30),  # Set all margins to zero
        )
        fig.update_yaxes(
            tickmode='linear', 
            tick0=0, 
            showgrid=False, 
            title_standoff=0,  # Distance between axis title and tick labels
            ticklabelposition="inside",  # Position tick labels inside to save space
        )
        
        # Remove y-axis numbers
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
        
#----Topic Representative Docs-----------------------------
numRepDocs = 3
if(selectedAlgo == 'BERTopic'):
    with st.expander("**Representative Documents**"):
        # st.header('')
        for topic_id, docs in representativeDocs.items():
            if topic_id == -1:
                continue
            st.subheader(f"Topic {topic_id + 1}:")
            for doc in docs[:numRepDocs]:
                st.write(doc)


st.markdown("---")
st.subheader("Topic Analysis")

#----Additional BERT Info-----------------------------------------
if(selectedAlgo == 'BERTopic'):
    st.write('\n')
    unassignedCount = len(df[df['Topic'] == -1])
    st.write(f'**Unassigned Documents: {unassignedCount}**')
    
#--Documents Per Topic Chart---------------------------------
dfAssignedDocs = df[df['Topic'] >= 0]
topic_document_count_df = dfAssignedDocs.groupby('Topic').size().reset_index(name='Document Count')
topic_document_count_df['Topic'] += 1
topic_document_count_df['Topic'] = 'Topic ' + topic_document_count_df['Topic'].astype(str)

fig = px.bar(
    topic_document_count_df, 
    x='Topic', 
    y='Document Count',
    color = 'Topic',
    color_discrete_sequence=topicColors,
)
fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=0),
    showlegend=False,
    title={
        'text': f'Documents Per Topic',
    },
)

st.plotly_chart(fig, use_container_width=True)
        
#----Topic Spread box Plot----------------------------------------
probColumn = None
if(selectedAlgo == 'BERTopic'):
    probColumn = 'Probs'
st.write('\n')
st.markdown('<p style="font-weight: bold;"> Distribution of Document Correlations Per Topic.</p>', help = 'Describes how correlated the assigned documents are to each topic. A longer/taller box implies a greater distinction between the documents in the topic. A higher box implies a closer association of the documents with the assigned topic', unsafe_allow_html=True)
st.plotly_chart(GetTopicDocumentStats(dfTopicDistributions, numTopics, topicColors, probColumn), use_container_width=True)

st.markdown("---")
st.subheader("Time Series Analysis")
#--Documents Over Time Chart---------------------------------
docFig = CreateDocTimeFig(df, numTopics, topicColors)
st.plotly_chart(docFig)

#--Documents Over Time Normalized Chart---------------------------------
docFig = CreateDocTimeFig(df, numTopics, topicColors, normalize=True)
st.plotly_chart(docFig)

if(selectedAlgo == 'LDA' or selectedAlgo == 'NMF'):
    st.markdown("---")
    st.subheader("Topic and Document Maps")
#---Topic Map---------------------------------
if(selectedAlgo != 'BERTopic'):
    ldaComponents = topicExtractionModel.components_
    reducedTopicWordMatrix = TSNE(n_components=2, perplexity=numTopics-1).fit_transform(topicExtractionModel.components_)

    documentMapHeight = 400
    fig = px.scatter(
        x=reducedTopicWordMatrix[:, 0], 
        y=reducedTopicWordMatrix[:, 1],
        title="Topic Map",)

    totalCount = topic_document_count_df['Document Count'].sum()
    topic_document_count_df['Percent'] = topic_document_count_df['Document Count'] / totalCount

    # Map colors based on the values of df['Topic']
    fig.update_traces(
        marker=dict(
            color=topicColors,
            colorscale='Viridis',
            size= 400 * topic_document_count_df['Percent'],
        )
    )

    # Remove axis labels
    fig.update_layout(
        xaxis_title='', 
        yaxis_title='', 
        height = documentMapHeight,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    # Remove y-axis numbers
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        tickmode='linear', 
        tick0=0, 
        showgrid=False, 
        title_standoff=0,  # Distance between axis title and tick labels
    )

    plot = st.plotly_chart(fig, use_container_width=True)

#-------------------------------------------------------------
if(selectedAlgo != 'BERTopic'):
    # Document Map
    # reducedTopicDistributions = PCA(n_components=2).fit_transform(documentTopicDistributions)
    reducedTopicDistributions = TSNE(n_components=2).fit_transform(documentTopicDistributions)

    documentMapHeight = 400
    fig = px.scatter(x=reducedTopicDistributions[:, 0], 
                    y=reducedTopicDistributions[:, 1], 
                    title="Document Map",)

    # Map colors based on the values of df['Topic']

    fig.update_traces(
        marker=dict(
            color=df['Topic'].map(dict(zip(range(numTopics), topicColors))),
            colorscale='Viridis'
        )
    )

    # Remove axis labels
    fig.update_layout(
        xaxis_title='', 
        yaxis_title='', 
        height = documentMapHeight,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    # Remove y-axis numbers
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(
        tickmode='linear', 
        tick0=0, 
        showgrid=False, 
        title_standoff=0,  # Distance between axis title and tick labels
    )

    st.plotly_chart(fig, use_container_width=True)

#----Raw Data----------------------------------------
displayDF = GetDataFrame(dataFileName, textColumnName, dateColumnName)
displayDF = displayDF[(displayDF[dateColumnName] >= pd.to_datetime(start_date)) & (displayDF[dateColumnName] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))]
st.markdown("---")
st.subheader("Raw Data")
displayDF['Topic'] = df['Topic'] + 1
st.write(displayDF)