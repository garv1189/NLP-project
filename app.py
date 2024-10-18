import streamlit as st
from src.summarization import summarize_text
from src.evaluation import evaluate_summaries
from src.nlp_utils import preprocess_text, extract_keywords, generate_wordcloud
from textblob import TextBlob

def analyze_sentiment(text):
    """Analyze sentiment of the input text using TextBlob."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Get the polarity (-1 to 1)
    subjectivity = blob.sentiment.subjectivity  # Get the subjectivity (0 to 1)
    return polarity, subjectivity

# Streamlit App Layout
st.title("Text Summarization and Analysis")

# Input Section
st.sidebar.header("Input Options")

input_method = st.sidebar.selectbox("Select input method:", ["Manual Text Input", "Upload Text File", "Use Example Text"])

# Initialize variables
input_text = ""

if input_method == "Manual Text Input":
    input_text = st.text_area("Enter text to summarize and analyze:", height=200)
elif input_method == "Upload Text File":
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")
elif input_method == "Use Example Text":
    input_text = """Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate goal of NLP is to enable computers to understand, interpret, and generate human languages in a way that is both valuable and meaningful. Applications of NLP include speech recognition, machine translation, sentiment analysis, and text summarization, among others. NLP has become a vital part of many applications, ranging from virtual assistants to translation services. As the field evolves, more sophisticated methods like deep learning and transformer models are being applied to improve performance and accuracy in various NLP tasks."""

# Process if text is available
if st.sidebar.button("Summarize"):
    if input_text:
        processed_text = preprocess_text(input_text)

        # Summarization
        summary = summarize_text(processed_text)

        # Keyword Extraction
        keywords = extract_keywords(processed_text)

        # Sentiment Analysis
        polarity, subjectivity = analyze_sentiment(processed_text)

        # Display Results
        st.markdown("### Summary:")
        st.write(summary)

        st.markdown("### Keywords:")
        st.write(", ".join(keywords))

        st.markdown("### Sentiment Analysis:")
        st.write(f"**Polarity:** {polarity:.2f} (Range: -1 to 1)\n**Subjectivity:** {subjectivity:.2f} (Range: 0 to 1)")

        # Word Cloud Section
        st.markdown("### Word Cloud:")
        wordcloud_fig = generate_wordcloud(processed_text)
        st.pyplot(wordcloud_fig)

        # Reference Summary for Evaluation
        reference_summary = st.text_area("Enter Reference Summary for Evaluation (optional):", height=100)

        if reference_summary:
            evaluation_scores = evaluate_summaries(reference_summary, summary)
            st.markdown("### Evaluation Metrics:")
            st.write("**ROUGE-1 Score:**", evaluation_scores['rouge-1'])
            st.write("**ROUGE-2 Score:**", evaluation_scores['rouge-2'])
            st.write("**ROUGE-L Score:**", evaluation_scores['rouge-l'])

        # Download Summary
        st.download_button("Download Summary", summary, file_name="summary.txt")
    else:
        st.error("Please provide some text input.")


