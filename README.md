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
Flagged Words for Most Recent Time Interval:
Seperates the documents into the time interval selected in the side bar options then used a modified version of TF-IDF called cTF-IDF to analyze the the words in each time interval. The words with the highest cTF-IDF scores are considered "Flagged". The chart dispalys the count of these words for each previous and the current time interval.

