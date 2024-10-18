from sklearn.metrics import accuracy_score
from rouge import Rouge  # Make sure to install `rouge`

def evaluate_summaries(reference, generated):
    """Evaluate the generated summary against the reference summary using ROUGE."""
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference, avg=True)
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f']
    }

def evaluate_accuracy(reference, generated):
    """Calculate accuracy based on exact matches."""
    return accuracy_score(reference, generated)
