import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from PrepareData import GetDataFrame

import plotly.graph_objects as go

# def PlotOverTime(ax, grouped_df, lineColor): 
#     ax.plot(grouped_df['TimeInterval'], grouped_df['Count'], color=lineColor)
    
def CreateDocTimeFig(df, primaryColor, topicCount, topicColors):
    fig = go.Figure()

    # Create a list to store visibility status for each trace
    visible_traces = [True] * topicCount

    x_values = []  # List to store all x values
    y_values = []  # List to store all y values

    for i in range(topicCount):
        # Filter data for topic i
        grouped_df = df.loc[df['Topic'] == i]  

        # Group data frame by time interval and store number of rows for each month in Count
        grouped_df = grouped_df.groupby('TimeInterval').size().reset_index(name='Count')
        grouped_df['TimeInterval'] = grouped_df['TimeInterval'].dt.to_timestamp()  # Convert to datetime

        # Add trace for each topic with initial visibility set to True
        fig.add_trace(go.Scatter(
            x=grouped_df['TimeInterval'],
            y=grouped_df['Count'],
            mode='lines',
            name=f'Topic {i}',
            line=dict(color=topicColors[i]),
            visible=visible_traces[i],
            showlegend=False 
        ))

        # Append x and y values to the lists
        x_values.extend(grouped_df['TimeInterval'])
        y_values.extend(grouped_df['Count'])

    # Calculate x and y range
    x_range = [min(x_values), max(x_values)]
    y_range = [min(y_values), max(y_values)]

    topicBtnLabel = ''
    
    if(topicCount > 9):
        topicBtnLabel = ''

    # Create buttons to show each trace individually or all traces
    buttons = [{'label': f'{topicBtnLabel}{i+1}', 'method': 'update', 'args': [{'visible': [True if j == i else False for j in range(topicCount)]}]} for i in range(topicCount)]
    buttons.append({'label': 'Show All', 'method': 'update', 'args': [{'visible': [True] * topicCount}]})


    # Update layout with buttons and set x and y axis ranges
    fig.update_layout(
        xaxis=dict(
            tickangle=45,
            showgrid=True,
            range=x_range  # Set x axis range
        ),
        yaxis=dict(
            title='Document Count',
            showgrid=True,
            range=y_range  # Set y axis range
        ),
        title='Document Count By Topic Over Time',
        updatemenus=[{
            'type': 'buttons',
            'direction': 'right',
            'showactive': True,
            'buttons': buttons,
            'x': 0.5,  # Set the x position of the buttons to the center
            'y': 1.1,  # Set the y position of the buttons to be above the chart
            'xanchor': 'center',  # Anchor the x position to the center
            'yanchor': 'top',  # Anchor the y position to the top
        }],
    )

    return fig