from fire import Fire
import numpy as np 
import tiktoken
from transformers import AutoTokenizer
from scipy.stats import chisquare, ks_2samp, wasserstein_distance, entropy
from metrics import *

def load_jsonl(file):
    import json
    with open(file, 'r') as f:
        return [json.loads(line) for line in f]

def load_csv(file):
    df = pd.read_csv(file)
    return df.to_dict(orient='records')

def load_json(file):
    return json.load(file, "r")

class DatasetLoader:
    """Helper class to load datasets"""
    def __init__(self, file):
        self.file = file
        self.data = None
    
    def load_data(self):
        if self.file.endswith('.jsonl'):
            import json
            with open(self.file, 'r') as f:
                self.data = [json.loads(line) for line in f]
        elif self.file.endswith('.csv'):
            import pandas as pd
            self.data = pd.read_csv(self.file).to_dict(orient='records')
        elif self.file.endswith('.json'):
            import json
            self.data = json.load(open(self.file, 'r'))
        else:
            raise ValueError("Unsupported file format")
        return self.data

class TokenCounter:
    """Helper class to manage different tokenizers"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        
        # Try tiktoken first
        try:
            if any(name in model_name.lower() for name in ['gpt', 'text-davinci', 'ada']):
                self.tokenizer = tiktoken.encoding_for_model(model_name)
                self.type = 'tiktoken'
            else:
                raise ValueError("Not a tiktoken model")
        except (ValueError, KeyError):
            # Fall back to HuggingFace tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.type = 'huggingface'
            except Exception as e:
                raise ValueError(f"Could not load tokenizer for {model_name}: {str(e)}")
    
    def count_tokens(self, text):
        """Count tokens in the given text"""
        if self.type == 'tiktoken':
            return len(self.tokenizer.encode(text))
        else:  # huggingface
            return len(self.tokenizer.encode(text, add_special_tokens=True))

def ascii_histogram(data, bins=10, width=50):
    """
    Create an ASCII histogram
    
    Args:
        data: List of numbers
        bins: Number of bins
        width: Width of the histogram in characters
    """
    # Calculate bins
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)
    
    # Generate histogram
    result = []
    for count, bin_start, bin_end in zip(hist, bin_edges[:-1], bin_edges[1:]):
        bar_length = int(width * count / max_count)
        bar = '█' * bar_length
        result.append(f'{int(bin_start):4d}-{int(bin_end):<4d} | {bar:<{width}} | {count:3d}')
    
    return '\n'.join(result), hist, bin_edges


def calculate_text_stats(data, model, show_histogram=True, key='text'):
    """
    Calculate statistics about text lengths in a dataset of dictionaries
    
    Args:
        data: List of dictionaries, each containing a 'text' key
        show_histogram: Whether to show ASCII histogram
        
    Returns:
        dict: Statistics about text lengths
    """
    # Initialize tokenizer
    token_counter = TokenCounter(model)
    
    # Extract token lengths
    lengths = [token_counter.count_tokens("\n".join([e["content"] for e in record[key] if e["role"] == "assistant"])) for record in data]
    
    # Calculate statistics
    stats = {
        '#': len(lengths),
        'mean': np.mean(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'p90': np.percentile(lengths, 90),
        'p95': np.percentile(lengths, 95),
        'p99': np.percentile(lengths, 99)
    }
    
    # Round all values to 1 decimal place
    stats = {k: round(v, 2) for k, v in stats.items()}
    
    # Generate text output
    output = []
    output.append("Text Length Statistics:")
    output.append("-" * 20)
    for stat, value in stats.items():
        output.append(f"{stat:>4}: {value:<8.1f}")
    
    if show_histogram:
        output.append("\nLength Distribution:")
        output.append("-" * 20)
        result, hist, bins = ascii_histogram(lengths)
        output.append(result)
    
    return [stats, lengths], "\n".join(output)


def compare_length_distribution(lengths1, lengths2, bins=10):
    """
    Compare length distribution differences between two datasets
    
    Args:
        lengths1: list of lengths
        lengths2: list of lengths
        bins: number of bins for length histogram
    """
    min_len = min(min(lengths1), min(lengths2))
    max_len = max(max(lengths1), max(lengths2))

    # Create histograms
    hist1, _ = np.histogram(lengths1, bins=bins, range=(min_len, max_len))
    hist2, _ = np.histogram(lengths2, bins=bins, range=(min_len, max_len))

    # normalize histograms
    # assert sum(hist1) == sum(hist2), "Histograms have different number of elements"
    norm_hist1 = hist1 / np.sum(hist1) + 1e-6
    norm_hist2 = hist2 / np.sum(hist2) + 1e-6

    # Calculate chi-square test
    # chi_stat, chi_p_value = chisquare(f_exp=hist1, f_obs=hist2)
    
    # Calculate KS test
    # ks_stat, ks_p_value = ks_2samp(lengths1, lengths2)

    # Earth Mover's Distance
    # emd = wasserstein_distance(norm_hist1, norm_hist2)

    # KL divergence
    kl_div = entropy(pk=norm_hist1, qk=norm_hist2)
    print(f"KL divergence: {kl_div}")
    
    return {
        # 'chisquare': (chi_stat, chi_p_value),
        # 'ks': (ks_stat, ks_p_value),
        # 'emd': emd,
        'kl_div': kl_div
    }


def calculate_text_similarity(data1, key1, key2, data2=None, sim_metrics=["ngram", "rouge", "meteor", "mauve"]):
    """
    Calculate similarity between two texts using different metrics
    
    Args:
        data1 & key1: loaded list of dicts for key1
        data2 & key2: loaded list of dicts for key2
        metrics: List of metrics to use
    
    Returns:
        dict: Dictionary of similarity scores
    """
    text1, text2 = [], []
    if data2:
        for d1, d2 in zip(data1, data2):
            text1.append(d1[key1])
            text2.append(d2[key2])
    else:
        for d in data1:
            text1.append("\n".join([e["content"] for e in d[key1] if e["role"] == "assistant"]))
            text2.append("\n".join([e["content"] for e in d[key2] if e["role"] == "assistant"]))

    scores = {}
    if "ngram" in sim_metrics:
        scores['ngram_dist'] = calculate_ngram(sources=text1, generations=text2)
    if "rouge" in sim_metrics:
        scores['rouge'] = calculate_rouge(sources=text1, generations=text2, use_stemmer=True)
    if "meteor" in sim_metrics:
        scores['meteor'] = calculate_meteor(sources=text1, generations=text2)
    if "mauve" in sim_metrics:
        result = calculate_mauve(sources=text1, generations=text2, device_id=0, verbose=False, featurize_model_name="BioMistral-7B", max_text_length=3072)
        scores['mauve'] = {
            "mauve": result["mauve"],
            "frontier_integral": result["frontier_integral"]
        }
    
    return scores


def main(dataset1: str, dataset2=None, model='gpt2', key1='text', key2=None):
    """
    Get statistics for one or two dataset(s); if two datasets are provided, do chi-sqaure and KS test for distribution differences
    For single dataset, specify [dataset1, key1]
    For two datasets, either specify [dataset1, dataset2, key1, key2] or [dataset1, key1, key2]
    
    Args:
        dataset1: Path to the first dataset
        dataset2: optional, Path to the second dataset
        key1: Key-to-analyze in the first dataset
        key2: optional, Key-to-analyze in the second dataset or Second key-to-analyze in the same dataset
        model: Model name or path to tokenizer
    """
    if not dataset2 and not key2:
        # single dataset
        print("Analyzing single dataset...")
        data = DatasetLoader(dataset1).load_data()
        print(calculate_text_stats(data, model, key=key1)[1])
    else:
        # two datasets
        assert key2, "Must provide key2"
    
        if dataset2:
            print("Comparing two seperate datasets...")
            # two seperate datasets
            data1 = DatasetLoader(dataset1).load_data()
            data2 = DatasetLoader(dataset2).load_data()
            result1, output1 = calculate_text_stats(data1, model, key=key1)
            result2, output2 = calculate_text_stats(data2, model, key=key2)
        else:
            # two keys in the same dataset
            print("Comparing two keys in the same dataset...")
            data1 = DatasetLoader(dataset1).load_data()
            data2 = None
            result1, output1 = calculate_text_stats(data1, model, key=key1)
            result2, output2 = calculate_text_stats(data1, model, key=key2)
        
        # calculate distribution differences
        diff = compare_length_distribution(result1[1], result2[1])
        # calculate similarity
        sim = calculate_text_similarity(data1=data1, data2=data2, key1=key1, key2=key2, sim_metrics=["mauve"])
        # sim = {}

        # print all results
        for o1, o2 in zip(output1.split("\n"), output2.split("\n")):
            print(f"{o1} \t {o2}")

        print("\nText Length Distribution Comparison:")
        print("-" * 20)
        # print(f"Chi-Square Test: (statistic={round(diff['chisquare'][0], 4)}, pvalue={round(diff['chisquare'][1], 4)})")
        print(f"KL Divergence: {round(diff['kl_div'], 4)}")
        # print(f"Earth Mover's Distance: {round(diff['emd'], 4)}")
        # print(f"Kolmogorov-Smirnov Test: (statistic={round(diff['ks'][0], 4)}, pvalue={round(diff['ks'][1], 4)})")

        print("\nText Similarity Scores:")
        print("-" * 20)
        for metric, score in sim.items():
            print(f"{metric.capitalize()}: {score}")


if __name__ == "__main__":
    Fire(main)
