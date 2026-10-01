'''
Script to train a full fine-tuning model for terminology generation.
'''
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedModel,
)
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import random
from typing import List, Tuple
import json
import os
from tqdm import tqdm
# import logging
import argparse
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Set up logging
# logging.basicConfig(level=logging.INFO)

class TerminologyDataset(Dataset):
    def __init__(
        self, 
        tokenizer,
        tokenizer_max_length: int = 512, 
        terminology_lists: List[List[str]] = None, 
        embedding_model = None,
        path_to_embeddings: str = None,
        #  aggregation_method: str = 'average', 
        embedding_batch_size: int = 32,
        noise_std: float = 0.1
    ):
        
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        assert (terminology_lists and embedding_model) or path_to_embeddings, "Either terminology_lists & embedding_models OR path_to_embeddings must be provided."
        self.terminology_lists = terminology_lists
        self.embedding_model = embedding_model
        self.path_to_embeddings = path_to_embeddings
        # self.aggregation_method = aggregation_method
        self.embedding_batch_size = embedding_batch_size
        self.noise_std = noise_std

        if self.path_to_embeddings and os.path.exists(self.path_to_embeddings):
            # Load embeddings from file
            self._load_embeddings(self.path_to_embeddings)
        else:
            self._prepare_embeddings(self.terminology_lists, self.embedding_model)
        
        self.embedding_dim = self.embeddings.size(1) if self.embeddings is not None else None
        
    def _prepare_embeddings(self, 
                            terminology_lists: List[List[str]], 
                            embedding_model):
        """Prepare embeddings for the terminology lists"""
        if not terminology_lists or not embedding_model:
            raise ValueError("Terminology lists and embedding model must be provided.")
        
        # Prepare data
        self.embeddings = []
        self.target_texts = []

        print(f"[Embedding Preparation]\tProcessing {len(terminology_lists)} terminology lists...")
        for i in tqdm(range(0, len(terminology_lists), self.embedding_batch_size)):
            batch = ["\n".join(terms) for terms in terminology_lists[i:i + self.embedding_batch_size]]
            with torch.no_grad():
                # Encode batch
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    show_progress_bar=False, 
                    convert_to_tensor=True,
                    batch_size=self.embedding_batch_size
                )
            # Aggregate embeddings
            # if self.aggregation_method == 'average':
            #     aggregated_embedding = torch.mean(batch_embeddings, dim=0)
            # elif self.aggregation_method == 'max':
            #     aggregated_embedding = torch.max(batch_embeddings, dim=0).values
            # else:
            #     raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
            # Store aggregated embedding
            # self.embeddings = torch.cat((self.embeddings, aggregated_embedding.unsqueeze(0)), dim=0)
            self.embeddings.append(batch_embeddings)
            
        self.embeddings = torch.cat(self.embeddings, dim=0)
        self.target_texts = ["\n".join(terms) for terms in terminology_lists]
        print(f"[Embedding Preparation]\tGenerated {len(self.embeddings)} embeddings.")
        
        # Save embeddings to file
        if self.path_to_embeddings:
            self._save_embeddings(self.embeddings, self.target_texts, self.path_to_embeddings)
    
    def _save_embeddings(self, 
                         embeddings: List[torch.tensor], 
                         target_texts: List[str],
                         file_path: str):
        """Save embeddings to a file"""
        torch.save({
            'embeddings': embeddings,
            'target_texts': target_texts
        }, file_path)
        print(f"[Embedding Preparation]\tEmbeddings saved to {file_path}")
    
    def _load_embeddings(self, file_path: str) -> Tuple[List[torch.tensor], List[str]]:
        """Load embeddings from a file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embeddings file {file_path} does not exist.")
        
        data = torch.load(file_path)
        self.embeddings = data['embeddings'].detach().cpu() # (number of rows, 1048)
        self.target_texts = data['target_texts']
        print(f"[Embedding Preparation]\tEmbeddings loaded from {file_path}")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        # Get original embedding
        embedding = self.embeddings[idx].clone().detach()
        
        # Add noise
        noisy_embedding = embedding + torch.normal(mean=0, std=self.noise_std, size=embedding.size())
        
        # Tokenize target text
        target_text = self.target_texts[idx]
        tokens = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.tokenizer_max_length,
            padding='max_length',
            # padding=True,
            return_tensors='pt'
        )
        
        return {
            'terminology_embedding': noisy_embedding.clone().to(torch.float32),
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': tokens['input_ids'].squeeze()
        }

class FineTuneTerminologyModel(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        sft_method: str = "prompt",
        prompt_length: int = 0,
        model_name: str = 'gpt2-large',
        freeze_lm: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sft_method = sft_method
        self.prompt_length = prompt_length
        self.freeze_lm = freeze_lm
        
        # Load language model (NOT frozen for full fine-tuning)
        if model_name.startswith('gpt2'):
            self.lm = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(model_name)
        self.hidden_size = self.lm.config.n_embd
        
        # Freeze language model parameters if specified
        if freeze_lm:
            print("[Model Configuration]\tFreezing language model parameters for full fine-tuning.")
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            for param in self.lm.parameters():
                param.requires_grad = True
            print("[Model Configuration]\tLanguage model parameters will be fine-tuned.")
        
        # Set up prompt method
        if self.sft_method == "prompt":
            if self.prompt_length > 0:
                print(f"[Model Configuration]\tUsing prompt method with length {self.prompt_length}.")
                self.embedding_projection = nn.Sequential(
                    nn.Linear(embedding_dim, self.hidden_size * self.prompt_length),
                    nn.LayerNorm(self.hidden_size * self.prompt_length),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_size * self.prompt_length, self.prompt_length * self.hidden_size),
                    nn.Tanh()
                )
            else:
                raise ValueError("Prompt length must be greater than 0 for prompt method.")
        elif self.sft_method == "projection":
            print("[Model Configuration]\tUsing projection method for fine-tuning.")
            self.embedding_projection = nn.Sequential(
                nn.Linear(embedding_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        else:
            raise ValueError(f"Unsupported SFT method: {self.sft_method}")
        
        # Initialize projection layers
        for module in self.embedding_projection:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        
        # Add special tokens for terminology conditioning
        # self.start_token_id = self.lm.config.vocab_size  # Will be added to tokenizer
        
    def forward(
        self, 
        terminology_embedding, 
        input_ids=None, 
        attention_mask=None, 
        labels=None,
        generation_args=None
    ):
        batch_size = terminology_embedding.size(0)
        
        # Project terminology embedding to model hidden size
        projected_embedding = self.embedding_projection(terminology_embedding)
        
        if input_ids is not None:
            # Training mode: prepend projected embedding as first token
            input_embeds = self.lm.transformer.wte(input_ids)
            
            # Add projected embedding as first token embedding
            if self.sft_method == "prompt":
                prefix_length = self.prompt_length
            elif self.sft_method == "projection":
                prefix_length = 1
            else:
                raise ValueError(f"Unsupported SFT method: {self.sft_method}")
            
            projected_embedding = projected_embedding.view(batch_size, prefix_length, self.hidden_size)
            full_embeds = torch.cat([projected_embedding, input_embeds], dim=1)
            
            # Extend attention mask
            embedding_attention = torch.ones(batch_size, prefix_length, device=attention_mask.device)
            full_attention_mask = torch.cat([embedding_attention, attention_mask], dim=1)
            
            # Extend labels (ignore the embedding token in loss)
            if labels is not None:
                embedding_labels = torch.full((batch_size, prefix_length), -100, device=labels.device)
                full_labels = torch.cat([embedding_labels, labels], dim=1)
            else:
                full_labels = None
            
            return self.lm(
                inputs_embeds=full_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels
            )
        else:
            # Inference mode
            # projected_embedding = projected_embedding.unsqueeze(1)
            if self.sft_method == "prompt":
                # projected_embedding = projected_embedding.view(batch_size, self.prompt_length, self.hidden_size)
                prefix_length = self.prompt_length
            elif self.sft_method == "projection":
                # projected_embedding = projected_embedding.unsqueeze(1)
                prefix_length = 1
            else:
                raise ValueError(f"Unsupported SFT method: {self.sft_method}")
            projected_embedding = projected_embedding.view(batch_size, prefix_length, self.hidden_size)
            return self.lm.generate(
                inputs_embeds=projected_embedding,
                attention_mask = torch.ones(batch_size, prefix_length, device=projected_embedding.device),
                pad_token_id=self.lm.config.eos_token_id,
                # max_length=512,
                # num_return_sequences=1,
                # temperature=0.7,
                # do_sample=True,
                # repetition_penalty=1.1
                **(generation_args or {})
            )

class TerminologyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        terminology_embedding = inputs.pop('terminology_embedding')
        
        outputs = model(
            terminology_embedding=terminology_embedding,
            input_ids=inputs.get('input_ids'),
            attention_mask=inputs.get('attention_mask'),
            labels=inputs.get('labels')
        )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def _save(self, output_dir: str, state_dict=None):
        """
        Override save method to handle shared tensors
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if self.model.freeze_lm:
            # Save projection only
            # Extract only projection layer parameters
            projection_state_dict = {}
            for name, param in self.model.named_parameters():
                if 'embedding_projection' in name:
                    projection_state_dict[name] = param.clone()
            
            torch.save(projection_state_dict, os.path.join(output_dir, "projection.bin"))
            print(f"[Checkpoint Saving]\tProjection-only checkpoint saved to: {output_dir}")
        else:
            # Save full model
            # Save using PyTorch format to avoid shared tensor issues
            if state_dict is None:
                state_dict = self.model.state_dict()
            # Save model state dict using PyTorch format
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
            print(f"[Checkpoint Saving]\tModel checkpoint saved to: {output_dir}")
        
        # Save model config
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(output_dir)
        elif hasattr(self.model, 'lm') and hasattr(self.model.lm, 'config'):
            self.model.lm.config.save_pretrained(output_dir)


def evaluate_reconstruction(
    model, 
    test_dataset, 
    tokenizer, 
    device=None, 
    eval_batch_size=8,
    generation_args=None,
):
    """Evaluate reconstruction quality with batch processing"""
    model.eval()
    results = []
    
    num_samples = len(test_dataset)
    print(f"[Evaluation]\tEvaluating on {num_samples} samples with batch size {eval_batch_size}...")
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, eval_batch_size)):
            batch_embeddings = test_dataset[batch_start:batch_start + eval_batch_size]['terminology_embedding']
            if isinstance(batch_embeddings, list):
                batch_embeddings = torch.stack(batch_embeddings).to(device)
            else:
                batch_embeddings = batch_embeddings.to(device)
            # Ensure batch size does not exceed dataset length
            current_batch_size = len(batch_embeddings)
            if current_batch_size == 0:
                continue
            
            batch_target_texts = test_dataset.target_texts[batch_start:batch_start + eval_batch_size]
            
            # Generate batch sequences
            try:
                generated_ids = model(
                    terminology_embedding=batch_embeddings,
                    generation_args=generation_args or {}
                )
                # Decode all sequences in batch
                batch_generated_texts = tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )

                # batch_generated_texts = []
                # for i in range(current_batch_size):
                #     generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                #     batch_generated_texts.append(generated_text)
            except Exception as e:
                print(f"Generation failed for batch starting at {batch_start}: {e}")
                batch_generated_texts = [""] * current_batch_size
            
            # Process each sample in the batch
            for i in range(current_batch_size):
                original_text = batch_target_texts[i]
                generated_text = batch_generated_texts[i]
                
                # Parse terms
                generated_terms = set(line.strip() for line in generated_text.split('\n') if line.strip())
                original_terms = set(line.strip() for line in original_text.split('\n') if line.strip())

                # print(f"[Evaluation]\tOriginal: {original_terms} \nGenerated: {generated_terms}")
                
                # Calculate metrics
                if len(original_terms) > 0:
                    precision = len(generated_terms & original_terms) / len(generated_terms) if len(generated_terms) > 0 else 0
                    recall = len(generated_terms & original_terms) / len(original_terms)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    precision = recall = f1 = 0
                
                results.append({
                    'original': original_text,
                    'generated': generated_text,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'original_count': len(original_terms),
                    'generated_count': len(generated_terms)
                })
    
    return results

def load_terminology_data(data_path: str) -> List[List[str]]:
    """
    Load terminology data from file
    Expected format: JSON file with list of terminology lists
    Example: [["term1", "term2"], ["term3", "term4", "term5"], ...]
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} not found.")

    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Train a full fine-tuning model for terminology generation.")
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')

    # Load configuration file
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found.")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Configuration:{")
    for k, v in config.items():
        print(f"\t{k}: {v}")
    print("}")
    
    print("[Initialization]\tLoading tokenizer...")
    if "gpt2" in config['lm_model_name']:
        tokenizer = GPT2Tokenizer.from_pretrained(config['lm_model_name'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["lm_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    if config["train_data_path"] and config["validation_data_path"]:
        # Load data
        print("[Initialization]\tLoading terminology data...")
        train_lists, val_lists = load_terminology_data(config['train_data_path']), load_terminology_data(config['validation_data_path'])
        print(f"[Initialization]\tTraining on {len(train_lists)} lists, validating on {len(val_lists)} lists")
    
        # Initialize embedding models
        print("[Initialization]\tLoading embedding model...")
        embedding_model = SentenceTransformer(config['embedding_model_name'])
        embedding_model.to(device)
    elif config["train_embeddings_path"] and config["validation_embeddings_path"]:
        # Load precomputed embeddings
        print("[Initialization]\tWill load precomputed embeddings...")
        train_lists, val_lists = None, None
        embedding_model = None
    else:
        raise ValueError("Either train_data_path & validation_data_path or train_embeddings_path & val_embeddings_path must be provided in the config.")
    
    tokenizer_max_length = tokenizer.model_max_length - config.get('prompt_length', 1)
    
    # Create datasets
    print("[Initialization]\tCreating training dataset...")
    train_dataset = TerminologyDataset(
        tokenizer, 
        tokenizer_max_length=tokenizer_max_length,
        terminology_lists=train_lists,
        embedding_model=embedding_model,
        path_to_embeddings=config.get('train_embeddings_path', None), 
        embedding_batch_size=config['embedding_batch_size'],
        noise_std=config['noise_std'], 
    )
    print(f"[Initialization]\tTraining dataset size: {len(train_dataset)}")
    
    print("[Initialization]\tCreating validation dataset...")
    val_dataset = TerminologyDataset(
        tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        terminology_lists=val_lists,
        embedding_model=embedding_model,
        path_to_embeddings=config.get('validation_embeddings_path', None),
        embedding_batch_size=config['embedding_batch_size'],
        noise_std=config['noise_std'],
    )
    print(f"[Initialization]\tValidation dataset size: {len(val_dataset)}")

    assert train_dataset.embedding_dim == val_dataset.embedding_dim, "Embedding dimensions do not match between train and validation datasets."
    embedding_dim = train_dataset.embedding_dim
    print(f"[Initialization]\tEmbedding dimension: {embedding_dim}")

    # sys.exit(0)  # Exit early for debugging purposes

    # Initialize model
    print(f"[Initialization]\tInitializing model with {config['lm_model_name']}...")
    model = FineTuneTerminologyModel(
        embedding_dim=embedding_dim,
        model_name=config['lm_model_name'],
        freeze_lm=config["freeze_lm"],
        sft_method=config["sft_method"],
        prompt_length=config["prompt_length"]
    )
    model.to(device)
    if config["lm_checkpoint_path"]:
        print(f"[Initialization]\tLoading model checkpoint from {config['lm_checkpoint_path']}...")
        if os.path.exists(config['lm_checkpoint_path']):
            if config["freeze_lm"]:
                # Load only projection weights
                projection_state_dict = torch.load(config['lm_checkpoint_path'], map_location=device)
                # Load only projection parameters
                model_dict = model.state_dict()
                projection_dict = {k: v for k, v in projection_state_dict.items() if k in model_dict}
                model_dict.update(projection_dict)
                model.load_state_dict(model_dict)
            else:
                # Load full model state dict
                print("[Initialization]\tLoading full model state dict...")
                model.load_state_dict(torch.load(config['lm_checkpoint_path'], map_location=device))
        else:
            raise FileNotFoundError(f"Model checkpoint {config['lm_checkpoint_path']} not found.")
    
    # Train
    if config["do_train"]:
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Initialization]\tTotal parameters: {total_params:,}")
        print(f"[Initialization]\tTrainable parameters: {trainable_params:,}")
        
        # Training arguments optimized for multi-GPU
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            overwrite_output_dir=True,
            num_train_epochs=config['num_train_epochs'],
            per_device_train_batch_size=config['train_batch_size'],
            # per_device_eval_batch_size=config['eval_batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            warmup_ratio=config['warmup_ratio'],
            logging_steps=config['logging_steps'],
            # eval_steps=config['eval_steps'],
            # save_steps=config['save_steps'],
            # evaluation_strategy=config['eval_strategy'],
            save_strategy=config['save_strategy'],
            # save_total_limit=5,
            # load_best_model_at_end=True,
            # metric_for_best_model="eval_loss",
            # greater_is_better=False,
            
            # Multi-GPU optimizations
            dataloader_num_workers=config['dataloader_num_workers'],
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,
            fp16=config["fp16"],  # Enable mixed precision for faster training
            
            # Memory optimizations
            # gradient_checkpointing=True,
            remove_unused_columns=False,
            
            # Reporting
            report_to=None,
            # save_total_limit=config["save_total_limit"],  # Keep only 3 best checkpoints
        )
        
        # Initialize trainer
        trainer = TerminologyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            # eval_dataset=val_dataset,
            tokenizer=tokenizer,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print("[Training]\tStarting training...")
        print(f"[Training]\tUsing {torch.cuda.device_count()} GPUs")
        trainer.train()
        
        # Save model
        print("[Training]\tSaving model...")
        trainer.save_model()
        tokenizer.save_pretrained(config['output_dir'])
        
        # Save config
        with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[Training]\tModel saved to: {config['output_dir']}")
    
    # Evaluate
    if config["do_eval"]:
        print("[Evaluating]\tEvaluating model...")    
        generation_args = {
            'max_length': config['generation_max_length'],
            'num_return_sequences': config['generation_num_return_sequences'],
            'do_sample': config['generation_do_sample'],
            'temperature': config['generation_temperature'],
            # 'top_k': config['generation_top_k'],
            'top_p': config['generation_top_p'],
            'repetition_penalty': config['generation_repetition_penalty'],
        }
        results = evaluate_reconstruction(model, val_dataset, tokenizer, device, eval_batch_size=config['eval_batch_size'], generation_args=generation_args)
        
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

        # save results to file
        output_file = os.path.join(config['output_dir'], 'eval_predictions.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # save statistics to file
        stats_file = os.path.join(config['output_dir'], 'eval_statistics.json')
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

        
        # Show sample results
        # print(f"\nSample Results:")
        # for i, result in enumerate(results[:5]):
        #     print(f"\n--- Sample {i+1} ---")
        #     print(f"Original ({result['original_count']} terms):")
        #     print(result['original'][:200] + "..." if len(result['original']) > 200 else result['original'])
        #     print(f"\nGenerated ({result['generated_count']} terms):")
        #     print(result['generated'][:200] + "..." if len(result['generated']) > 200 else result['generated'])
        #     print(f"Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}")

if __name__ == "__main__":
    main()