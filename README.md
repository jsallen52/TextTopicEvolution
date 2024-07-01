# Text Topic Evolution App

This application visualizes the evolution of topics in textual data using NMF and LDA topic extraction techniques.

## Setup

1. Clone the repository.
2. Install the required dependencies by navigating to the cloned folder and running `pip install -r dependencies.txt`.
3. Install NLTK data for parts of speech model.
    - Run python interpreter and type commands:
    - import nltk
    - nltk.download()
    - see https://www.nltk.org/data.html# for more details
4. Run the Streamlit app by navigating to the cloned folder and executing `streamlit run MainApp.py`.
5. Open the app in your browser at `http://localhost:8501`.

## Additional Chart Info
**Flagged Words for Most Recent Time Interval:**
Seperates the documents into the time interval selected in the side bar options then used a modified version of TF-IDF called cTF-IDF('https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html') to analyze the the words in each time interval. The words with the highest cTF-IDF scores are considered "Flagged". The chart dispalys the count of these words for each previous and the current time interval.

**Topic Word Charts**
Each chart displays the top n (number can be slected in sidebar options) most descriptive words for each topic. This is found using the **Topic-Term Matrix** which describes the probability for each word being a part of each topic. For LDA and NMF algorithms this matrix is calculated in parallel with the **Document-Topic Matrix** which respresents the probability for each document belonging to each Topic. For BERTopic the **Topic-Term Matrix** is caclulated usign the cTF-IDF algorithm post clustering the documents into topics.
