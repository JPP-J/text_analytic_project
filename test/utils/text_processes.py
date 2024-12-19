# utils/text_processes.py
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF
from sklearn.feature_extraction.text import CountVectorizer #TO, T F

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text processes tokenization, lower cases, stopword and stemming
def preprocess_text(text):
    # Handle cases where text is NaN, None, or not a string
    if not isinstance(text, str):
        return []  # Return an empty list for non-string or NaN values

    # Tokenization
    words = word_tokenize(text)

    # Lowercase, remove stopwords, and stem words and not  use stop word when it's contains other not letter
    words = [stemmer.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]

    return words  # Return processed words as a list

# Text processes only tokenization and lower cases
def tokenize_text(text):
    # Handle cases where text is NaN, None, or not a string
    if not isinstance(text, str):
        return []  # Return an empty list for non-string or NaN values

    # Tokenization
    words = word_tokenize(text.lower())

    return words  # Return processed words as a list


def load_vectorize_text(path, col_x='data', col_y='labels', type_vector='tfidf'):
    # Part1: Load dataset
    df = pd.read_csv(path)

    # --------------------------------------------------------------------------------------
    # Part3: text processing
    # Apply the preprocessing function to the 'text' column : tokenization, lower cases, stopword and stemming
    df['processed_words'] = df[col_x].apply(preprocess_text)
    # print(df[['data', 'processed_words']][0:5])

    # Apply the preprocessing function to the 'text' column : tokenization and lower cases - optional
    df['tokenize_words'] = df[col_x].apply(tokenize_text)
    # print(df[['data', 'tokenize_words']][0:5])

    # --------------------------------------------------------------------------------------
    # Text Vectorize
    df['processed_words'] = df['processed_words'].apply(lambda x: " ".join(x))
    x = df['processed_words']  # column to clustering
    y = df[col_y]

    # Step 1: Vectorize the processed text using vectorize
    # Check for different vectorizer types and apply accordingly
    if type_vector == 'to':
        to_vectorizer = CountVectorizer()  # or pruning > CountVectorizer(min_df=2, max_df=0.95)
        X_to = to_vectorizer.fit_transform(x)
        return X_to , y# return or further process as needed
    elif type_vector == 'tf':
        tf_vectorizer = TfidfVectorizer(use_idf=False)  # Set use_idf=False for pure TF
        X_tf = tf_vectorizer.fit_transform(x)
        return X_tf, y
    elif type_vector == 'tfidf':
        tfidf_vectorizer = TfidfVectorizer()
        X_tfidf = tfidf_vectorizer.fit_transform(x)
        return X_tfidf, y
    else:
        raise ValueError(f"Unsupported vectorizer type: {type_vector}")



