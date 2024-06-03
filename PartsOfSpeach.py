from nltk import word_tokenize, pos_tag

def filter_nouns(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Apply part-of-speech tagging
    tagged_words = pos_tag(words)
    # Extract and return only nouns
    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return ' '.join(nouns)


def FilterForNouns(df, textColumnName):
    df[textColumnName] = df[textColumnName].apply(filter_nouns)
    return df