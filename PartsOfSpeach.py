from nltk import word_tokenize, pos_tag

def filter_nouns(text):
    """
    Filter for nouns in the given text.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text containing only the nouns.
    """
    # Tokenize the text
    words = word_tokenize(text)
    # Apply part-of-speech tagging
    tagged_words = pos_tag(words)
    # Extract and return only nouns
    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return ' '.join(nouns)


def FilterForNouns(df, textColumnName):
    """
    Apply filter_nouns to the text column of a pandas DataFrame and return the DataFrame with the updated text column.

    Args:
        df (pandas.DataFrame): The DataFrame to apply filter_nouns to.
        textColumnName (str): The name of the text column in the DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with the updated text column with only nouns.
    """
    df[textColumnName] = df[textColumnName].apply(filter_nouns)
    return df
