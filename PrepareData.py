import pandas as pd

def GetFullDataFrame(fileName):
    """
    Read a DataFrame from a CSV or JSON file based on the file extension.
    
    Args:
        fileName (str): The name of the file to read
        
    Returns:
        pd.DataFrame: The DataFrame read from the file
    """
    if(fileName.endswith('.json')):
        df = pd.read_json(fileName)
    else:
        df = pd.read_csv(fileName)
    return df

def GetDataFrame(fileName, textColumnName, dateColumnName, filterOutColumns = True):
    """
    Read a DataFrame from a CSV or JSON file, select specific columns, and process the date column.
    
    Args:
        fileName (str): The name of the file to read
        textColumnName (str): The name of the text column
        dateColumnName (str): The name of the date column
        
    Returns:
        pd.DataFrame: The processed DataFrame with text and date columns
    """
    if(fileName.endswith('.json')):
        df = pd.read_json(fileName)
    else:
        df = pd.read_csv(fileName)
    
    if(filterOutColumns):
        df = df[[dateColumnName, textColumnName]]
    #drop any columns that do not have a date or text
    df = df.dropna(subset=[textColumnName, dateColumnName], inplace=False) 
    df[dateColumnName] = pd.to_datetime(df[dateColumnName]).dt.tz_localize(None)
    return df

def AssignTimeInterval(df, dateColumnName, timeInterval):
    """
    Assign time intervals to the DataFrame based on the specified time interval.
    
    Args:
        df (pd.DataFrame): The input DataFrame
        dateColumnName (str): The name of the date column
        timeInterval (str): The desired time interval ('Monthly', 'Yearly', etc.)
        
    Returns:
        pd.DataFrame: The DataFrame with assigned time intervals as an additional column named 'TimeInterval'
    """
    intervalChar = 'W'
    if(timeInterval == 'Monthly'):
        intervalChar = 'M'
    if(timeInterval == 'Yearly'):
        intervalChar = 'Y'
    df['TimeInterval'] = df[dateColumnName].dt.to_period(intervalChar)
    return df
