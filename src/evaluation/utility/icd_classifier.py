import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import json, random
from typing import List, Dict, Any
import os
import argparse
from tqdm import tqdm


ICD9_LABELS = [
    "",
    "Infectious And Parasitic Diseases",
    "Neoplasms",
    "Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders",
    "Diseases Of The Blood And Blood-Forming Organs",
    "Mental Disorders",
    "Diseases Of The Nervous System And Sense Organs",
    "Diseases Of The Circulatory System",
    "Diseases Of The Respiratory System",
    "Diseases Of The Digestive System",
    "Diseases Of The Genitourinary System",
    "Complications Of Pregnancy, Childbirth, And The Puerperium",
    "Diseases Of The Skin And Subcutaneous Tissue",
    "Diseases Of The Musculoskeletal System And Connective Tissue",
    "Congenital Anomalies",
    "Certain Conditions Originating In The Perinatal Period",
    "Symptoms, Signs, And Ill-Defined Conditions",
    "Injury And Poisoning",
    "External Causes Of Injury And Poisoning",
    "Factors Influencing Health Status And Contact With Health Services"
]

ICD10_LABELS = [
    "",
    "Certain Infectious And Parasitic Diseases",
    "Neoplasms",
    "Diseases Of The Blood And Blood-Forming Organs And Certain Disorders Involving The Immune Mechanism",
    "Endocrine, Nutritional And Metabolic Diseases",
    "Mental And Behavioural Disorders",
    "Diseases Of The Nervous System",
    "Diseases Of The Eye And Adnexa",
    "Diseases Of The Ear And Mastoid Process",
    "Diseases Of The Circulatory System",
    "Diseases Of The Respiratory System",
    "Diseases Of The Digestive System",
    "Diseases Of The Skin And Subcutaneous Tissue",
    "Diseases Of The Musculoskeletal System And Connective Tissue",
    "Diseases Of The Genitourinary System",
    "Pregnancy, Childbirth And The Puerperium",
    "Certain Conditions Originating In The Perinatal Period",
    "Congenital Malformations, Deformations And Chromosomal Abnormalities",
    "Symptoms, Signs And Abnormal Clinical And Laboratory Findings, Not Elsewhere Classified",
    "Injury, Poisoning And Certain Other Consequences Of External Causes",
    "External Causes Of Morbidity And Mortality",
    "Factors Influencing Health Status And Contact With Health Services",
    "Codes For Special Purposes"
]

COMB_LABELS = [
    'Certain Infectious And Parasitic Diseases', 
    'Neoplasms', 
    'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders', 
    'Diseases Of The Blood And Blood-Forming Organs And Certain Disorders Involving The Immune Mechanism', 
    'Mental And Behavioural Disorders', 
    'Diseases Of The Nervous System And Sense Organs', 
    'Diseases Of The Circulatory System', 
    'Diseases Of The Respiratory System', 
    'Diseases Of The Digestive System', 
    'Diseases Of The Genitourinary System', 
    'Complications Of Pregnancy, Childbirth, And The Puerperium', 
    'Diseases Of The Skin And Subcutaneous Tissue', 
    'Diseases Of The Musculoskeletal System And Connective Tissue', 
    'Congenital Malformations, Deformations And Chromosomal Abnormalities', 
    'Certain Conditions Originating In The Perinatal Period', 
    'Symptoms, Signs And Abnormal Clinical And Laboratory Findings, Not Elsewhere Classified', 
    'Injury, Poisoning And Certain Other Consequences Of External Causes', 
    'External Causes Of Morbidity And Mortality, Injusy and Poisoning', 
    'Factors Influencing Health Status And Contact With Health Services', 
    'Diseases Of The Eye And Adnexa', 
    'Diseases Of The Ear And Mastoid Process', 
    'Codes For Special Purposes'
]

def load_and_preprocess_data(data_path: str, label2idx: dict, synth_data_file: str="", text_column: str = "text", label_column: str = "labels"):
    """
    Load data and preprocess for multi-label classification
    
    Args:
        data_path: Path to file
        text_column: Name of column containing clinical notes
        label_columns: List of column names for ICD codes (binary 0/1)
    
    Returns:
        texts: List of clinical note texts
        labels: List of binary label arrays
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        df = df.fillna("")
        data = df.to_dict(orient='records')
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError("Unsupported file format for files with golden labels. Please provide a CSV or JSONL file.")

    if synth_data_file:
        synth_data = json.load(open(synth_data_file, "r"))
        source2synth = {}
        for record in synth_data:
            source_text = "".join([e["content"] for e in record["original_messages"] if e["role"] == "assistant"]).strip("\n")
            synth_text = "".join([e["content"] for e in record["messages"] if e["role"] == "assistant"])
            source2synth[source_text] = synth_text
    else:
        source2synth = None
    
    processed_data = []
    for entry in data:
        _labels = entry[label_column].split("|")
        tlabel = [0.] * len(label2idx)
        for _l in _labels:
            tlabel[label2idx[_l]] = 1.
        text = entry[text_column].strip("\n")
        if source2synth:
            assert text in source2synth, f"{entry} not found in synthetic corpus"
        processed_data.append({
            text_column: source2synth[text] if source2synth else entry[text_column],
            label_column: tlabel
        })
    return processed_data

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # print("input shape: ", inputs["input_ids"].shape)
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Use BCEWithLogitsLoss for multi-label classification
        loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def precision_at_k(y_true, y_pred_proba, k):
    """
    Calculate precision@k for multi-label classification.
    
    Args:
        y_true: numpy array of shape (n_samples, n_labels) with binary labels (0 or 1)
        y_pred_proba: numpy array of shape (n_samples, n_labels) with predicted probabilities
        k: int, number of top predictions to consider
    
    Returns:
        float: precision@k score
    """
    n_samples = y_true.shape[0]
    precision_scores = []
    
    for i in range(n_samples):
        # Get indices of top k predictions (highest probabilities)
        top_k_indices = np.argsort(y_pred_proba[i])[-k:]
        
        # Count how many of the top k predictions are correct
        correct_predictions = np.sum(y_true[i][top_k_indices])
        
        # Precision@k for this sample
        precision_at_k_sample = correct_predictions / k
        precision_scores.append(precision_at_k_sample)
    
    # Return average precision@k across all samples
    return np.mean(precision_scores)


def compute_metrics(predictions, labels):
    """
    Compute multi-label classification metrics
    """
    # Convert probabilities to binary predictions (threshold = 0.5)
    y_pred = (predictions > 0.5).int().numpy()
    y_true = labels.astype(int)
    
    # Compute metrics
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # AUC scores using probabilities (not binary predictions)
    # try:
    #     auc_micro = roc_auc_score(y_true, probs.numpy(), average='micro')
    #     auc_macro = roc_auc_score(y_true, probs.numpy(), average='macro')
    # except ValueError:
    #     # Handle case where some classes might not be present in validation set
    #     auc_micro = 0.0
    #     auc_macro = 0.0
    auc_micro = roc_auc_score(y_true, predictions.numpy(), average='micro')
    auc_macro = roc_auc_score(y_true, predictions.numpy())

    # Exact match accuracy (all labels must be correct)
    exact_match = accuracy_score(y_true, y_pred)

    # precision@k
    prec_3 = precision_at_k(y_true, predictions.numpy(), k=3)
    prec_5 = precision_at_k(y_true, predictions.numpy(), k=5)

    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'auc_micro': auc_micro,
        'auc_macro': auc_macro,
        'exact_match': exact_match,
        'prec@3': prec_3,
        'prec@5': prec_5
    }



if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Train a multi-label classifier for ICD codes.")
    parser.add_argument('--train_file', type=str, default="", help='Path to training data CSV file with golden ICD labels')
    parser.add_argument('--test_file', type=str, default="", help='Path to testing data CSV file with golden ICD labels')
    parser.add_argument('--train_synth_file', type=str, default="", help='Path to synthetic data jsonl file without ICD labels')
    parser.add_argument('--model_name', type=str, default='yikuan8/Clinical-Longformer', help='Pre-trained model name')
    parser.add_argument('--max_length', type=int, default=None, help='Maximum sequence length for tokenization')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for optimizer')
    parser.add_argument('--output_dir', type=str, default='ICD_output', help='Directory to save the model and logs')
    parser.add_argument('--fold', type=int, default=0, help="Seed")

    args = parser.parse_args()
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Logging] Using device: {device}")

    # Load ICD labels based on version
    assert args.train_file or args.test_file, "Please provide a train or test file."
    if (args.train_file and "comb" in args.train_file) or (args.test_file and "comb" in args.test_file):
        ICD_LABELS = COMB_LABELS
    elif (args.train_file and "ICD9" in args.train_file) or (args.test_file and "ICD9" in args.test_file):
        ICD_LABELS = ICD9_LABELS
    elif (args.train_file and "ICD10" in args.train_file) or (args.test_file and "ICD10" in args.test_file):
        ICD_LABELS = ICD10_LABELS
    else:
        raise ValueError("Invalid ICD version. Choose either 'ICD9' or 'ICD10' or 'comb'.")
    label2idx = {label: idx for idx, label in enumerate(ICD_LABELS)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    print(f"[Logging] ICD labels: {ICD_LABELS}")
    
    # Prepare model and tokenizer
    print(f"[Logging] Loading model and tokenizer...")
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ICD_LABELS),
        problem_type="multi_label_classification",
        # ignore_mismatched_sizes=True  # This allows loading with different head
    )
    model.to(device)
    print(f"[Logging] Model and tokenizer loaded successfully.")
    
    args.max_length = args.max_length if args.max_length else config.max_position_embeddings - 2
    print(f"[Logging] Model max lenght is {args.max_length}")

    # Train
    if args.train_file:
        # Load and preprocess data
        def tokenize_function(example):
            inputs = tokenizer(example['text'], truncation=True, max_length=args.max_length)
            inputs['labels'] = example['labels']
            return inputs
        print(f"[Logging] Loading and preprocessing training data from {args.train_file}...")
        train_raw_data = load_and_preprocess_data(args.train_file, label2idx, args.train_synth_file,)
        random.shuffle(train_raw_data)
        print(f"[Logging] Number of training samples: {len(train_raw_data)}")
        # print a sample of the training data
        print(f"[Logging] Sample training data: {train_raw_data[0]}")
        train_dataset = Dataset.from_list(train_raw_data)
        # Tokenize the training data
        train_dataset = train_dataset.map(tokenize_function)

        # Initialize trainer
        training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=10,
        warmup_ratio=0.1,
        save_strategy="no"
        )
        trainer = MultiLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            # compute_metrics=compute_metrics,
        )
        # Train the model
        print(f"[Logging] Starting training...")
        trainer.train()
        print(f"[Logging] Training completed.")
        # Save the model
        # print(f"[Logging] Saving model...")
        # trainer.save_model(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)
        # print(f"[Logging] Model saved to {args.output_dir}")
    
    # Evaluate
    if args.test_file:
        print(f"[Logging] Loading and preprocessing validation data from {args.test_file}...")
        test_raw_data = load_and_preprocess_data(args.test_file, label2idx)
        print(f"[Logging] Number of testing samples: {len(test_raw_data)}")

        model.eval()
        test_outputs = []
        test_pred_probs = torch.empty(0, len(idx2label))
        test_golden_labels = []
        for entry in tqdm(test_raw_data):
            inputs = tokenizer(entry['text'], truncation=True, max_length=args.max_length, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.Sigmoid()(torch.Tensor(logits.cpu()))
                test_pred_probs = torch.cat([test_pred_probs, probs], dim=0)
                test_golden_labels.append(entry["labels"])

                entry_pred_labels, entry_golden_labels = [], []
                pred = (probs[0] > 0.5).int().numpy()
                for i, p in enumerate(pred):
                    if p == 1:
                        entry_pred_labels.append(idx2label[i])
                for i, p in enumerate(entry["labels"]):
                    if p == 1.:
                        entry_golden_labels.append(idx2label[i])

                # save results
                test_outputs.append({
                    "text": entry["text"],
                    "labels": "|".join(entry_golden_labels),
                    "pred_labels": "|".join(entry_pred_labels),
                    "pred": "|".join([str(v) for v in pred])
                })
        
        # compute metrics
        results = compute_metrics(test_pred_probs, np.array(test_golden_labels))
        print(f"=============Average metrics==============")
        for key, value in results.items():
            print(key,": ", round(value, 4))
        
        # Save predictions to file
        output_df = pd.DataFrame(test_outputs)
        output_df.to_csv(os.path.join(args.output_dir, f"fold{args.fold}_predictions_epoch{args.num_epochs}.csv"), index=False)
        print(f"[Logging] Predictions saved to {os.path.join(args.output_dir, f'fold{args.fold}_predictions_epoch{args.num_epochs}.csv')}")
        # Save average metrics to file
        with open(os.path.join(args.output_dir, f'fold{args.fold}_metrics_epoch{args.num_epochs}.json'), 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[Logging] Metric saved to {os.path.join(args.output_dir, f'fold{args.fold}_metrics_epoch{args.num_epochs}.json')}")

        # cross validation

        # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python icd_classifier.py --train_file "data/folds/train_comb_fold2.csv" --test_file "data/folds/test_comb_fold2.csv" --train_synth_file "Llama-3.2-1B-Instruct-SFT-DP_dist-EP8/synthesized_data_dp_dist_ep8_multistep_1b_rep1.2_freq1.0_seed0.json" --model_name yikuan8/Clinical-Longformer --num_epochs 30 --output_dir output_icd_CV/ep8 --fold 2

        # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python icd_classifier.py --train_file "data/folds/train_comb_fold5.csv" --test_file "data/folds/test_comb_fold5.csv" --train_synth_file "Llama-3.2-1B-Instruct-SFT-DP_dist-EP8/synthesized_data_dp_dist_ep8_multistep_1b_rep1.2_freq1.0_seed0.json" --model_name yikuan8/Clinical-Longformer --num_epochs 30 --output_dir output_icd_CV/ep8 --fold 5

        # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python icd_classifier.py --train_file "data/folds/train_comb_fold1.csv" --test_file "data/folds/test_comb_fold1.csv" --train_synth_file "Llama-3.2-1B-Instruct-SFT/synthesized_data_multistep_1b_rep1.2_lb6.0_seed42.json" --model_name yikuan8/Clinical-Longformer --num_epochs 30 --output_dir output_icd_CV/sft --fold 1

        # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python icd_classifier.py --train_file "data/folds/train_comb_fold1.csv" --test_file "data/folds/test_comb_fold1.csv" --train_synth_file "Llama-3.2-1B-Instruct-SFT-DP_dist-EP5/synthesized_data_dp_dist_ep5_multistep_1b_rep1.2_freq0.4_lb1.0_seed777.json" --model_name yikuan8/Clinical-Longformer --num_epochs 30 --output_dir output_icd_CV/sft --fold 1
    
        # dryrun testing code
        # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python icd_classifier.py --test_file "data/folds/test_comb_fold1.csv" --model_name yikuan8/Clinical-Longformer --num_epochs 5 --output_dir output_icd/