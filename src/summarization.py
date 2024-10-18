from transformers import pipeline

def summarize_text(text, max_length=130):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']
