import pandas as pd

def GetFullDataFrame(fileName):
    if(fileName.endswith('.json')):
        df = pd.read_json(fileName)
    else:
        df = pd.read_csv(fileName)
    return df

def GetDataFrame(fileName, textColumnName, dateColumnName):
    # df = pd.read_csv(fileName)
    
    if(fileName.endswith('.json')):
        df = pd.read_json(fileName)
    else:
        df = pd.read_csv(fileName)
    
    df = df[[dateColumnName, textColumnName]]
    df = df.dropna(subset=[textColumnName, dateColumnName], inplace=False) 
    df[dateColumnName] = pd.to_datetime(df[dateColumnName]).dt.tz_localize(None)
    return df

def AssignTimeInterval(df,dateColumnName, timeInterval):
    # Label rows based on time interval
    intervalChar = 'W'
    if(timeInterval == 'Monthly'):
        intervalChar = 'M'
    if(timeInterval == 'Yearly'):
        intervalChar = 'Y'
    df['TimeInterval'] = df[dateColumnName].dt.to_period(intervalChar)
    return df