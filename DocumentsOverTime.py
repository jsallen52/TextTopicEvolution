import pandas as pd
import plotly.graph_objects as go
    
def CreateDocTimeFig(df: pd.DataFrame, topicCount: int, topicColors: list[str], normalize: bool = False) -> go.Figure:
    """
    Create a line graph of document counts over time per topic either normalized using percentage of docuemnts for each time interval or simple count.

    Args:
        df (pd.DataFrame): DataFrame containing document information with a 'TimeInterval' column and a 'Topic' column.
        topicCount (int): Number of topics.
        topicColors (List[str]): List of colors for each topic.
        normalize (bool, optional): If True, normalize the counts by the total count per month. Defaults to False.

    Returns:
        go.Figure: Plotly figure object for the line graph.
    """
    # Determine the title of the figure
    figTitle = 'Document Count Over Time Per Topic'
    if normalize:
        figTitle = 'Percentage of Documents Over Time Per Topic'

    # Create an empty Plotly figure
    fig = go.Figure()

    # Update the layout of the figure
    fig.update_layout(
        title=figTitle
    )

    # Create a list to store visibility status for each trace
    visible_traces = [True] * topicCount

    x_values = []  # List to store all x values
    y_values = []  # List to store all y values

    # Calculate the total count per month
    monthTotals = df[df['Topic'] >= 0].groupby('TimeInterval').size().reset_index(name='Count')['Count']

    # Iterate over each topic
    for i in range(topicCount):
        # Filter data for topic i
        grouped_df = df.loc[df['Topic'] == i]  

        # Group data frame by time interval and store number of rows for each month in Count
        grouped_df = grouped_df.groupby('TimeInterval').size().reset_index(name='Count')
        grouped_df['TimeInterval'] = grouped_df['TimeInterval'].dt.to_timestamp()  # Convert to datetime

        if normalize:
            grouped_df['Count'] = grouped_df['Count'] / monthTotals

        # Add trace for each topic with initial visibility set to True
        fig.add_trace(go.Scatter(
            x=grouped_df['TimeInterval'],
            y=grouped_df['Count'],
            mode='lines',
            name=f'Topic {i + 1}',
            line=dict(color=topicColors[i]),
            visible=visible_traces[i],
            showlegend=True 
        ))

        # Append x and y values to the lists
        x_values.extend(grouped_df['TimeInterval'])
        y_values.extend(grouped_df['Count'])

    return fig
