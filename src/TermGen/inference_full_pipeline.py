'''
Script to run the full inference pipeline for term generation.
Including embedding and generation steps.
'''
import json
import os
import typing
from tqdm import tqdm
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, AutoTokenizer
from tqdm import tqdm

import argparse

from dprp_embeddings import DPRPEmbeddings
from train import FineTuneTerminologyModel

import random
# set seed for reproducibility
random.seed(42)

def prepare_terminology_lists(data_path):
    terms_data = json.load(open(data_path, 'r'))
    # Extract the terminology lists from the terms data
    terminology_lists = []
    for nid, sections in terms_data.items():
        note_terms = []
        for section, content in sections.items():
            if section == "full_note":
                continue
            note_terms.append(content["terms"])
        terminology_lists.append(note_terms)
    return terminology_lists

class TerminologyDataset(Dataset):
    def __init__(
        self, 
        file_path,
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_batch_size=32,
        dprp_args=None,
        device="cpu",
    ):
        self.note_ids, self.section_names, self.terminology_lists = self._load_data(file_path)
        print(f"[TerminologyDataset]\t Example: {self.note_ids[22]}-{self.section_names[22]}\n{self.terminology_lists[22]}")
        self.device = device
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to(device)
        self.embedding_batch_size = embedding_batch_size
        self._embed()  # Embed the terminology lists
        assert len(self.note_ids) == self.embeddings.shape[0], "Number of note IDs must match number of embeddings"
        print(f"[TerminologyDataset]\t Example: {self.embeddings[22]}")
        self.dprp_args = dprp_args
        self.dp_info = self._apply_dprp()  # Apply DPRP if arguments are provided
        print(f"[TerminologyDataset]\t Example: {self.dp_embeddings[22]}")
        print(f"[TerminologyDataset]\t DP info: {self.dp_info}")

    def _load_data(self, file_path):
        print(f"[TerminologyDataset]\t Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            terms_data = json.load(f)
        self.original_data = terms_data  # Store original data for reference
        note_id, section_name, terminology_list = [], [], []
        for nid, sections in terms_data.items():
            for section, content in sections.items():
                if section == "full_note":
                    continue
                note_id.append(nid)
                section_name.append(section)
                # sec_terms = list(set(t.lower() for t in content["terms"]))
                sec_terms = list(dict.fromkeys([t.lower() for t in content["terms"]]))
                sec_terms = sec_terms if sec_terms else [""]
                terminology_list.append(sec_terms)
        print(f"[TerminologyDataset]\t Loaded {len(note_id)} notes with {len(section_name)} sections.")
        return note_id, section_name, terminology_list
    
    def _embed(self):
        print("[TerminologyDataset]\t Embedding terminology lists...")
        embeddings = []
        for i in tqdm(range(0, len(self.terminology_lists), self.embedding_batch_size)):
            batch = ["\n".join(terms) for terms in self.terminology_lists[i:i+self.embedding_batch_size]]
            with torch.no_grad():
                # Encode batch
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    show_progress_bar=False, 
                    convert_to_tensor=True,
                    batch_size=len(batch),
                )
            embeddings.append(batch_embeddings)
        self.embeddings = torch.cat(embeddings, dim=0).to(self.device)
        print(f"[TerminologyDataset]\t Generated embeddings of shape: {self.embeddings.shape}")

    # Apply DPRP for differential privacy
    def _apply_dprp(self):
        if self.dprp_args:
            print("[TerminologyDataset]\t Applying DPRP for differential privacy...")
            dprp = DPRPEmbeddings(**self.dprp_args)
            dp_embeddings, dp_info = dprp.release_embeddings_modified(self.embeddings)
            self.dp_embeddings = torch.tensor(dp_embeddings, device=self.device, dtype=torch.float32)
            print(f"[TerminologyDataset]\t DPRP embeddings shape: {self.dp_embeddings.shape}")
            return dp_info
        else:
            print("[TerminologyDataset]\t No DPRP arguments provided, skipping DPRP.")
            self.dp_embeddings = self.embeddings
            return None

    def __len__(self):
        return len(self.dp_embeddings)

    def __getitem__(self, idx):
        return {
            "terminology_embedding": self.dp_embeddings[idx].clone().to(torch.float32).to(self.device)
        }


def generate_terms(
    model,
    test_dataset,
    tokenizer,
    device=None,
    eval_batch_size=8,
    generation_args=None,
):
    """Evaluate reconstruction quality with batch processing"""
    model.eval()
    pred_terminology_lists, results = [], []
    
    num_samples = len(test_dataset)
    print(f"[Evaluation]\t Evaluating on {num_samples} samples with batch size {eval_batch_size}...")

    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, eval_batch_size)):
            batch_embeddings = test_dataset.dp_embeddings[batch_start:batch_start + eval_batch_size]
            batch_embeddings = batch_embeddings.to(device)
            
            current_batch_size = batch_embeddings.shape[0]
            if current_batch_size == 0:
                continue

            # Generate batch sequence
            generated_ids = model(
                terminology_embedding=batch_embeddings,
                generation_args=generation_args or {}
            )
            # Decode generated sequences
            generated_texts = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            for i in range(current_batch_size):
                original_terms = test_dataset.terminology_lists[batch_start + i]
                generated_terms = [t.strip() for t in generated_texts[i].split("\n") if t.strip()]

                # parse terms
                original_terms_set = set(original_terms)
                generated_terms_set = set(generated_terms)

                # calculate metrics
                if len(original_terms_set) > 0:
                    precision = len(original_terms_set & generated_terms_set) / len(generated_terms_set) if generated_terms_set else 0.0
                    recall = len(original_terms_set & generated_terms_set) / len(original_terms_set) if original_terms_set else 0.0
                    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                else:
                    precision, recall, f1 = 0.0, 0.0, 0.0
                
                pred_terminology_lists.append(generated_terms)
                results.append({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "original_count": len(original_terms_set),
                    "generated_count": len(generated_terms_set)
                })
    return pred_terminology_lists, results

            

def main():
    parser = argparse.ArgumentParser(description="Inference on fine-tuned TermGen.")
    parser.add_argument('--config', type=str, default='inference_config.json', help='Path to configuration file')

    # Load configuration file
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found.")
    with open(args.config, 'r') as f:
        config = json.load(f)

    file_path = config["input_file_path"]
    embedding_model_name = config["embedding_model_name"]  # Example model, can be changed
    embedding_batch_size = config["embedding_batch_size"]
    
    # DP parameters
    dp_epsilon = config["dp_epsilon"]/6.0
    dp_delta = config["dp_delta"]/6.0
    dp_budget_allocation = config["dp_budget_allocation"]
    
    # fine-tuned model parameters
    generation_lm_model = config["generation_lm_model"]  # Example model, can be changed
    generation_ckpt_path = config["generation_ckpt_path"]  # Path to the fine-tuned model checkpoint
    generation_max_length = config["generation_max_length"]  # Example max length for generation
    generation_temperature = config["generation_temperature"]  # Example temperature for generation
    generation_top_p = config["generation_top_p"]  # Example top-p for generation
    generation_repetition_penalty = config["generation_repetition_penalty"]  # Example repetition penalty for generation
    generation_batch_size = config["generation_batch_size"]  # Batch size for generation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # change dprp_args to {} if you want to disable differential privacy
    if config.get("disable_differential_privacy", False):
        dprp_args = {}
    else:
        dprp_args = {
            'epsilon': dp_epsilon,
            'delta': dp_delta,
            'budget_allocation': dp_budget_allocation
        }
    
    # Prepare dataset
    print("[Main]\t =========================================\n[Main]\t Preparing terminology dataset...\n[Main]\t =========================================")
    test_dataset = TerminologyDataset(
        file_path=file_path,
        embedding_model_name=embedding_model_name,
        embedding_batch_size=embedding_batch_size,
        dprp_args=dprp_args,
        device=device
    )

    # Infer the terms from trained model
    # clear all torch cache
    torch.cuda.empty_cache()
    print("[Main]\t =========================================\n[Main]\t Initializing FineTuneTerminologyModel...\n[Main]\t =========================================")
    model = FineTuneTerminologyModel(
        embedding_dim=test_dataset.dp_embeddings.shape[1],
        model_name=generation_lm_model,
        freeze_lm=False,
        sft_method="projection",
        prompt_length=1
    )
    model.to(device)
    # load the model checkpoint
    print("[Main]\t Loading generation model checkpoint...")
    model.load_state_dict(torch.load(generation_ckpt_path, map_location=device))
    model = model.float()

    if "gpt2" in generation_lm_model:
        tokenizer = GPT2Tokenizer.from_pretrained(generation_lm_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(generation_lm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # generation
    print("[Main]\t =========================================\n[Main]\t Generating terms...\n[Main]\t =========================================")
    generation_args = {
        "max_length": generation_max_length,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": generation_temperature,
        "top_p": generation_top_p,
        "repetition_penalty": generation_repetition_penalty,
    }
    pred_terminology_lists, results = generate_terms(
        model,
        test_dataset,
        tokenizer,
        device=device,
        eval_batch_size=generation_batch_size,
        generation_args=generation_args,
    )

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    
    # Detailed statistics
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    print(f"\nDetailed Statistics:")
    print(f"Precision - Mean: {np.mean(precisions):.3f}, Std: {np.std(precisions):.3f}")
    print(f"Recall - Mean: {np.mean(recalls):.3f}, Std: {np.std(recalls):.3f}")
    print(f"F1 - Mean: {np.mean(f1s):.3f}, Std: {np.std(f1s):.3f}")
    
    print(f"\nPerfect reconstructions: {sum(1 for r in results if r['f1'] == 1.0)}/{len(results)}")
    print(f"High quality (F1 > 0.8): {sum(1 for r in results if r['f1'] > 0.8)}/{len(results)}")
    
     # save statistics to file
    stats_file = os.path.join(output_dir, config["output_stats_file"])
    stats = {
        'average_precision': avg_precision,
        'average_recall': avg_recall,
        'average_f1': avg_f1,
        'perfect_reconstructions': sum(1 for r in results if r['f1'] == 1.0),
        'high_quality_reconstructions': sum(1 for r in results if r['f1'] > 0.8),
        'num_samples': len(results)
    }
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # save generated terms to file
    original_data = test_dataset.original_data
    for note_id, section_name, pred_terms in zip(test_dataset.note_ids, test_dataset.section_names, pred_terminology_lists):
        if note_id not in original_data:
            print(f"[Warning]\t Note ID {note_id} not found in original data, skipping...")
        original_data[note_id][section_name]["terms"] = pred_terms
    output_file = os.path.join(output_dir, config["output_predictions_file"])
    with open(output_file, 'w') as f:
        json.dump(original_data, f, indent=4)


if __name__ == "__main__":
    main()