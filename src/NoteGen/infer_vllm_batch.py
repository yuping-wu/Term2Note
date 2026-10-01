import json
import argparse
import time
import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import Dataset, load_from_disk
import accelerate
from tqdm import tqdm
from transformers import AutoTokenizer


def load_jsonl(file_path):
    """Load data from a jsonl file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_evaluation_loop(model_name, dataset_path, output_path, max_tokens=2048, lora=None, stop_tokens=None, tensor_parallel=1, num_samples=-1, seed=0):
    """
    Run a vLLM evaluation loop that processes clinical discharge note generation.

    Uses a HuggingFace dataset and generates responses for each conversation,
    processing all conversations round by round and letting vLLM handle batching.
    """
    # First ensure the dataset exists, otherwise prepare it
    if not os.path.exists(dataset_path):
        print(f"Dataset doesn't exist at {dataset_path}, please prepare it first.")
        return

    # Load the dataset
    print(f"Loading dataset from {dataset_path}")
    # eval_dataset = load_from_disk(dataset_path)# 
    formatted_data = load_jsonl(dataset_path)
    eval_dataset = Dataset.from_list(formatted_data)
    if num_samples > -1:
        eval_dataset = eval_dataset.select(range(min(num_samples, len(load_from_disk(dataset_path)))))

    # Load the model and tokenizer
    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name, dtype="bfloat16", tensor_parallel_size=tensor_parallel, enable_lora=True if lora else False, max_lora_rank=64 if lora else 16)
    lora_request = LoRARequest("medical_adapter", 1, lora) if lora else None
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,  # deterministic for consistent output
        top_p=1.0,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] + (stop_tokens or []),  # Assuming this is "<|im_end|>" or equivalent
        repetition_penalty=1.2,
        frequency_penalty=0.4,
        # presence_penalty=1.0,
        # min_p=0.7,
        logit_bias={tokenizer.eos_token_id: 2.0},
        seed=seed
    )
    print(f"Inferring with sampling params:\n{sampling_params}")
    
    # Initialize conversation histories for all examples
    all_conversations = []
    for idx in range(len(eval_dataset)):
        sample = eval_dataset[idx]
        original_messages = sample.get("messages", [])
        if not original_messages:
            print(f"Warning: No messages found in sample {idx}, skipping")
            continue
        all_conversations.append({"id": idx, "messages": [], "original_messages": original_messages})

    # Process all conversations round by round
    start_time = time.time()

    # Find maximum rounds across all conversations
    max_rounds = max([len(conv["original_messages"]) for conv in all_conversations])

    # Track which conversations need processing at each round
    active_conversations = list(range(len(all_conversations)))

    round_idx = 0
    while round_idx < max_rounds and active_conversations:
        print(f"Processing round {round_idx+1}/{max_rounds}")
        batch_inputs = []
        batch_indices = []

        # Collect inputs for this round
        for conv_idx in active_conversations[:]:
            conversation = all_conversations[conv_idx]
            original_messages = conversation["original_messages"]
            generated_messages = conversation["messages"]

            # Check if we've reached the end of this conversation
            if round_idx >= len(original_messages):
                active_conversations.remove(conv_idx)
                continue

            current_message = original_messages[round_idx]

            # Add current message to conversation history
            if current_message["role"] != "assistant":
                generated_messages.append(current_message)
            # if round_idx > 1:
            #     print(generated_messages)
            #     assert False
            # If this is a user message, prepare it for inference
            if current_message["role"] == "user":
                # Apply chat template to format the conversation
                formatted_prompt = tokenizer.apply_chat_template(
                    generated_messages, tokenize=False, add_generation_prompt=True
                )
                batch_inputs.append(formatted_prompt)
                batch_indices.append(conv_idx)
            elif round_idx + 1 < len(original_messages) and original_messages[round_idx + 1]["role"] == "assistant":
                # If next message is assistant in original, skip it in our generated version
                # (we'll generate our own assistant message)
                round_idx += 1

        # If we have inputs, process them
        if batch_inputs:
            # Let vLLM handle batching internally
            outputs = llm.generate(
                prompts=batch_inputs, sampling_params=sampling_params, use_tqdm=True, lora_request=lora_request
            )

            # Process outputs and add to conversation histories
            for j, output in enumerate(outputs):
                conv_idx = batch_indices[j]
                response_text = output.outputs[0].text.strip()

                # Create and add assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    # "prompt": batch_inputs[j]
                }
                all_conversations[conv_idx]["messages"].append(assistant_message)
                # print(assistant_message)
                # Debug info for first example
                if j == 0:
                    print(f"Sample response: {response_text[:100]}...")

        # Move to next round
        round_idx += 1

    # Extract final conversations for output
    final_results = []
    for conversation in all_conversations:
        final_results.append(
            {
                "id": conversation["id"],
                "messages": conversation["messages"],
                "original_messages": conversation["original_messages"],
            }
        )

    # Save all generated conversations
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"Evaluation completed in {total_time:.2f}s")
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation for discharge note generation")
    parser.add_argument("--model", type=str, required=True, help="Name or path of the model to load")
    parser.add_argument("--lora", type=str, required=True, help="Name or path of the lora adapters")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HuggingFace dataset")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for results")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument("--num-gpus", type=int, default=1, help="Num GPUs to use")
    parser.add_argument("--stop-tokens", nargs='+', type=int, default=None, help="Tokens to stop on.")
    parser.add_argument("--num-samples", type=int, default=-1, help="Number of samples to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    args = parser.parse_args()
    args.lora = None if args.lora == "None" else args.lora
    args.stop_tokens = [128001, 128008] # Llama
    # args.stop_tokens = [106] # Gemma
    run_evaluation_loop(args.model, args.dataset, args.output, args.max_tokens, args.lora, args.stop_tokens, args.num_gpus, args.num_samples, args.seed)


if __name__ == "__main__":
    main()
