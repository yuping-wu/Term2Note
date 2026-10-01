'''
Script to compute perplexity of a list of text snippets using a pre-trained language model.
'''

import json, os, re
import argparse
import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import evaluate
from evaluate import logging


def extract_seed(filename):
    match = re.search(r'seed\d+', filename)
    if match:
        return match.group()
    return False

def match_predictions(pred_file_path, pred_file_pattern):
    pred_files = [f for f in os.listdir(pred_file_path) if f.startswith(pred_file_pattern) and f.endswith('.json')]
    if not pred_files:
        raise ValueError(f"No files found matching the pattern {pred_file_pattern} in {pred_file_path}.")
    source2pred, seed_order = {}, []
    for idx, pred_file in enumerate(pred_files):
        full_path = os.path.join(pred_file_path, pred_file)
        seed = extract_seed(pred_file)
        if not seed:
            raise ValueError(f"Cannot identify seed in {pred_file}.")
        seed_order.append(seed)
        print(f"Loading dataset from {full_path}")
        predictions = json.load(open(full_path, "r"))
        for pred in predictions:
            original_messages = pred.get("original_messages", [])
            if not original_messages:
                raise ValueError(f"No 'original_messages' found in prediction {pred}.")
            source = ""
            for message in original_messages:
                if message.get("role") == "assistant":
                    source += message.get("content", "")
            if source not in source2pred:
                source2pred[source] = {
                    "original_messages": original_messages,
                }
            
            pred_message = pred.get("messages", [])
            if not pred_message:
                raise ValueError(f"No 'messages' found in prediction {pred}.")
            # add prediction
            source2pred[source][f"messages_{idx}"] = pred_message
    assert len(source2pred) == 204, f"Expected 204 predictions, but found {len(source2pred)}. Please check the input files."
    return source2pred, seed_order


def agg_result(dataset, order = ["seed42", "seed777", "seed66", "seed0"]):
    ppl = {
        k: [] for k in order
        # k: [] for k in order if k.startswith("seed")
    }

    for data in dataset:
        perplexities = data["perplexities"]
        assert len(perplexities) == len(order), f"Expected {len(order)} perplexities, got {len(perplexities)}"
        for i in range(len(order)):
            seed = order[i]
            # if i != 2 and i != 4:
            ppl[seed].append(perplexities[i])

    # Calculate mean and standard deviation for each seed
    mean_ppl = {k: np.mean(v) for k, v in ppl.items()}
    std_ppl = {k: np.std(v) for k, v in ppl.items()}
    # Print results
    print("Perplexity Results:")
    for seed in order:
        if seed in mean_ppl:
            print(f"{seed}: Mean = {mean_ppl[seed]:.2f}, Std = {std_ppl[seed]:.2f}")

    # report average perplexity across all seeds, and standard deviation
    avg_all_ppl = np.mean(list(mean_ppl.values()))
    print(f"Average Perplexity across all seeds: {avg_all_ppl:.2f}")  
    std_all_ppl = np.std(list(mean_ppl.values()))
    print(f"Standard Deviation of Perplexity across all seeds: {std_all_ppl:.2f}")


_CITATION = """"""

_DESCRIPTION = """"""

_KWARGS_DESCRIPTION = """
Args:
    model_id (str): model used for calculating Perplexity
    predictions (list of str): input text, each separate text snippet
        is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
    device (str): device to run on, defaults to 'cuda' when available
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.

"""

# @evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Perplexity(evaluate.Metric):
    def __init__(self, model_id, device=None):
        super(evaluate.Metric, self).__init__()
        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(
        self, predictions, batch_size: int = 16, add_start_token: bool = True,  max_length=None
    ):
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = self.tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in range(0, len(encoded_texts), batch_size):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute perplexity for a list of text snippets.")
    parser.add_argument("--model", type=str, required=True, help="Path or identifier of the pre-trained model.")
    parser.add_argument(
        "--predictions_file_path", type=str, default=None, help="Path to the input file containing text snippets (jsonl format)."
    )
    parser.add_argument(
        "--predictions_file_pattern", type=str, required=True, help= "Path to the input file containing text snippets (jsonl format)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for processing text snippets."
    )
    parser.add_argument(
        "--add_start_token", action="store_true", help="Whether to add the start token to the texts."
    )
    parser.add_argument(
        "--max_length", type=int, default=None, help="Max input length"
    )
    parser.add_argument(
        "--output_file", type=str, default="perplexity_results.json", help="Output file to save the results."
    )

    args = parser.parse_args()
    if args.predictions_file_path is None:
        args.predictions_file_path = args.model
    
    # Load predictions from the specified file
    metric = Perplexity(
        model_id=args.model,
        # device="cuda" if torch.cuda.is_available() else "cpu"
    )
    source2pred, seed_order = match_predictions(args.predictions_file_path, args.predictions_file_pattern)

    # Example usage
    all_results = []
    all_results_detailed = []
    for src, all_messages in tqdm(source2pred.items(), desc="Processing predictions"):
        # Extract the original messages
        original_messages = all_messages["original_messages"]
        # Extract the prediction messages
        predictions = []
        for key in [f"messages_{i}" for i in range(len(all_messages) - 1)]:
            cur_messages = all_messages[key]

            # note only
            pred = ["Discharge Summary:"]
            for message in cur_messages:
                if message["role"] == "assistant":
                    pred.append(message["content"])
            predictions.append("\n\n".join(pred))

        # Compute perplexity for the predictions
        results = metric.compute(
            predictions=predictions, 
            batch_size=args.batch_size if args.batch_size else len(predictions), 
            add_start_token=args.add_start_token,
            max_length=args.max_length,
        )
        perplexities = results["perplexities"]

        sel_perplexity_idx = np.argmin(perplexities)

        # Store the source and final prediction
        final_message = all_messages[f"messages_{sel_perplexity_idx}"]
        all_results.append(
            {
                "perplexity": perplexities[sel_perplexity_idx],
                "messages": final_message,
                "original_messages": original_messages,
            }
        )

        all_results_detailed.append(
            {
                "perplexities": perplexities,
                **all_messages
            }
        )
    
    # Save results to a JSON file
    output_file = os.path.join(args.predictions_file_path, args.output_file)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    output_detailed_file = os.path.join(args.predictions_file_path, args.output_file[:-8]+"detailed.json")
    with open(output_detailed_file, "w") as f:
        json.dump(all_results_detailed, f, indent=2)
    
    # aggregate result
    print("Aggregating results.....")
    agg_result(all_results_detailed, seed_order)