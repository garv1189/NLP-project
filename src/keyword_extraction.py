# src/keyword_extraction.py

def extract_keywords(text):
    """Extract keywords from the input text."""
    # Example using TF-IDF or any other method
    # This is a placeholder; replace with actual extraction logic

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])

    # Get feature names (words)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Get the highest scoring keywords based on TF-IDF
    tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
    top_n = 10  # Number of top keywords to extract

    # Extract top keywords
    top_keywords = feature_names[tfidf_sorting][:top_n]

    # Convert numpy array to list of strings
    return top_keywords.tolist()
