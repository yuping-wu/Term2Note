'''
Script containing the metrics used to evaluate the text pairs of source and synthesized text.
'''
import evaluate
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chisquare, ks_2samp, entropy, wasserstein_distance

def calculate_rouge(sources, generations, **kwargs):
    """
    Calculate ROUGE scores between source and generated text
    
    Args:
        sources: List of source texts
        generations: List of generated texts
        **kwargs: Additional keyword arguments to pass to ROUGE scorer, such as use_stemmer, please refer to https://huggingface.co/spaces/evaluate-metric/rouge for more information
        
    Returns:
        dict: Dictionary of ROUGE scores
    """
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=generations, references=sources, **kwargs)
    scores = {k: round(v * 100, 4) for k, v in scores.items()}
    return scores


def calculate_meteor(sources, generations, **kwargs):
    """
    Calculate METEOR scores between source and generated text
    
    Args:
        sources: List of source texts
        generations: List of generated texts
        **kwargs: Additional keyword arguments to pass to METEOR scorer, such as alpha, beta, gamma, please refer to https://huggingface.co/spaces/evaluate-metric/meteor for more information
        
    Returns:
        dict: Dictionary of METEOR scores
    """
    meteor = evaluate.load("meteor")
    scores = meteor.compute(predictions=generations, references=sources, **kwargs)
    scores = {k: round(v, 4) for k, v in scores.items()}
    return scores


def calculate_mauve(sources, generations, **kwargs):
    """
    Calculate MAUVE scores between source and generated text
    
    Args:
        sources: List of source texts
        generations: List of generated texts
        **kwargs: Additional keyword arguments to pass to MAUVE scorer, such as device_id, please refer to https://huggingface.co/spaces/evaluate-metric/mauve for more information
        
    Returns:
        dict: Dictionary of MAUVE scores
    """
    mauve = evaluate.load("mauve")
    scores = mauve.compute(predictions=generations, references=sources, **kwargs)
    scores = {
        "mauve": round(scores.mauve, 4),
        "frontier_integral": round(scores.frontier_integral, 4),
        # "divergence_curve": scores.divergence_curve,
        # "p_hist": scores.p_hist,
        # "q_hist": scores.q_hist
    }
    return scores


def vectorize(vectorizer, corpus):
    topk = 1000
    freq = vectorizer.fit_transform(corpus).toarray()
    feature_names = vectorizer.get_feature_names_out()
    # keep top k features and frequencies
    freq = freq.sum(axis=0) / len(corpus)
    if topk:
        topk_indices = freq.argsort()[-topk :]
    else:
        topk_indices = range(len(freq))
    return {
        feat: val
        for feat, val in zip(feature_names[topk_indices], freq[topk_indices])
    }

def calculate_ngram(sources, generations):
    result = {}
    ngram = [1, 2, 3]
    for n in ngram:
        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words="english", max_df=0.95, min_df=2)
        corpus_topk = vectorize(vectorizer, generations)
        reference_topk = vectorize(vectorizer, sources)
        corpus_freq, reference_freq = [], []
        for feat in set(corpus_topk.keys()) | set(reference_topk.keys()):
            corpus_freq.append(corpus_topk.get(feat, 1e-6))
            reference_freq.append(reference_topk.get(feat, 1e-6))
        # calculate KL divergence
        kl_div = entropy(pk=reference_freq, qk=corpus_freq)
        result[f"{n}-gram"] = float(kl_div)
    
    return result

