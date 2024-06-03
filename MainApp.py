from sklearn.decomposition import NMF, PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from PrepareData import AssignTimeInterval, GetDataFrame
from DocumentsOverTime import CreateDocTimeFig
from LDA import GetTopics

import plotly.express as px
import plotly.graph_objects as go

from PartsOfSpeach import FilterForNouns

additional_stop_words = []

# additional_stop_words= ['pilot', 'failure','control','resulted','loss','maintain','directional','flight','airplane','determined','reasons','available','based','contributing', 'improper', 'landing']

useAdditionalStopWords = True

dataFileName = '2020_2023.json'
#analysisNarrative
textColumnName = 'cm_probableCause'
dateColumnName = 'cm_eventDate'

# Create init dataframe
df = GetDataFrame(dataFileName, textColumnName, dateColumnName)

# Title of the app
st.title('Incident Documents Analysis')

#-------Filters Side Bar-----------------------------------
with st.sidebar.form(key='filter_form'):

    st.header('General Filters')
    
    start_date, end_date = st.date_input(
        "Select Date Range",
        value=(df[dateColumnName].min(), df[dateColumnName].max()),
        format="MM/DD/YYYY",
        key="date_range"
    )
    
    options = ['Weekly', 'Monthly', 'Yearly']
    selectedTimeInterval = st.selectbox('Time Interval', options, index = 1)
    
    topWordCount = st.slider('Top Words', 5, 50, 20)
    
    #-------Side Bar Parts of Speech
    st.subheader('Parts of Speech')
    # Filter for nouns only
    useNounsOnly = st.checkbox("Nouns Only", value=True, key="nouns_only_checkbox")
    
    #-------Side Bar Stop Words
    st.subheader('Stop Words')
    useStopWords = st.checkbox("Use Standard Stop Words", value=True, key="stop_words_checkbox")

    vectorizer = CountVectorizer(stop_words= 'english' if useStopWords else None)
    X = vectorizer.fit(df[textColumnName])

    useAdditionalStopWords = st.checkbox("Use Additional Stop Words", value=True, key="my_checkbox2", disabled=not useStopWords)

    additional_stop_words = st.multiselect('Select Additional Stop Words', list(vectorizer.get_feature_names_out()), default=additional_stop_words, disabled= (not useAdditionalStopWords) or (not useStopWords))

    if(not useStopWords):
        useAdditionalStopWords = False
        
    #-------------------------

    st.subheader('Topic Extraction')
    algoOptions = ['LDA', 'NMF']
    selectedAlgo= st.selectbox('Algorithm', algoOptions, index=1)
    
    numTopics = st.slider('Topic Count', 3, 20, 4)
    wordsPerTopic = st.slider('Words Per Topic', 8, 15, 10)
    
    # Submit button
    st.form_submit_button(label='Apply Filters')

#-------------------------------------------------------------
df = df[(df[dateColumnName] >= pd.to_datetime(start_date)) & (df[dateColumnName] <= pd.to_datetime(end_date))]

if(useNounsOnly):
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

documentCount = df.shape[0]

all_stop_words = list(CountVectorizer(stop_words='english').get_stop_words()) if useStopWords else None

if(useAdditionalStopWords):
    all_stop_words = all_stop_words + additional_stop_words

vectorizer = CountVectorizer(stop_words=all_stop_words, max_df=0.95, min_df=2)

docTermMatrix = vectorizer.fit_transform(df[textColumnName])
total_words = docTermMatrix.sum()
unique_words = len(vectorizer.get_feature_names_out())

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

#--LDA---------------------------------------------
useNMF = (selectedAlgo == 'NMF')

if(useNMF):
    #NMF used TFIDF for vectorization
    vectorizer = TfidfVectorizer(stop_words=all_stop_words, max_df=0.95, min_df=2)
    docTermMatrix = vectorizer.fit_transform(df[textColumnName])
    
if(not useNMF):
    topicExtractionModel = LatentDirichletAllocation(n_components=numTopics, max_iter=50, learning_method='online')
else:
     topicExtractionModel = NMF(n_components=numTopics, random_state=42)

documentTopicDistributions = topicExtractionModel.fit_transform(docTermMatrix)
docTopics = np.argmax(documentTopicDistributions, axis=1)
df['Topic'] = docTopics

topicDFfs = GetTopics(topicExtractionModel, vectorizer, wordsPerTopic)

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
    '#fffafa',
    ]

st.write('\n')

figHeights = 300
chartsPerRow = 4
for i in range(numTopics):
    if(i % chartsPerRow == 0):
        bar_cols= st.columns(chartsPerRow)
        
    with bar_cols[i % chartsPerRow]:
        fig = go.Figure(data=[go.Bar(x=topicDFfs[i]["Frequency"], y=topicDFfs[i]["Word"], orientation='h',  marker_color=topicColors[i])])
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
        
        
#--Documents Per Topic Chart---------------------------------
topic_document_count_df = df.groupby('Topic').size().reset_index(name='Document Count')
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
        'text': f'Estimated Documents Per Topic',
        'x': 0.5,
        'xanchor': 'center'
    },
)

st.plotly_chart(fig, use_container_width=True)

#-------------------------------------------------------------
# check boxes for each topic

primaryColor =  "#31333F"
docFig = CreateDocTimeFig(df, primaryColor, numTopics, topicColors)
st.plotly_chart(docFig)

#---Topic Map---------------------------------
ldaComponents = topicExtractionModel.components_
reducedTopicWordMatrix = TSNE(n_components=2, perplexity=3).fit_transform(topicExtractionModel.components_)

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
        color=df['Topic'].map(dict(zip(df['Topic'].unique(), topicColors))),
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
st.subheader("Raw Data")
st.markdown("---")
displayDF['Topic'] = df['Topic'] + 1
st.write(displayDF)