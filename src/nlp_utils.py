import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def preprocess_text(text):
    """Function to preprocess text (you can add more preprocessing steps as needed)."""
    # Example: converting to lowercase and stripping white spaces
    return text.lower().strip()


def extract_keywords(text, top_n=10):
    """Extract top N keywords from the input text using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])

    # Get feature names
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Sort by TF-IDF score
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]

    # Extract top N keywords
    top_keywords = feature_names[tfidf_sorting][:top_n]

    return top_keywords.tolist()


def generate_wordcloud(text):
    """Generate and return a word cloud."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  # Remove axes
    return fig
