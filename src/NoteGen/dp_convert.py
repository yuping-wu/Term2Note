# Now you can load the model normally with PyTorch or Hugging Face
import os
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("Llama-3.2-1B-Instruct")  # First load base architecture
# model = AutoModelForCausalLM.from_pretrained("gemma-3-1b-it")  # First load base architecture

# dp - Llama
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
# Path to the directory containing your DeepSpeed checkpoint files
checkpoint_dir = "Llama-3.2-1B-Instruct-SFT-DP-EP8/checkpoint-1412"
tag = "global_step1134"

# Path where you want to save the consolidated model
output_dir = checkpoint_dir
output_path = os.path.join(output_dir, "consolidated_model.pt")

# Convert the distributed checkpoint to a single consolidated checkpoint
state_dict = convert_zero_checkpoint_to_fp32_state_dict(
    checkpoint_dir,  # Directory containing bf16_zero_* and zero_pp_* files
    output_path,      # Output consolidated model path
    tag=tag
)

# Check what keys are present/missing
model_keys = set(k for k, _ in model.named_parameters())
state_dict_keys = set(state_dict.keys())
missing_keys = model_keys - state_dict_keys
print(f"Missing keys: {missing_keys}")


if "lm_head.weight" not in state_dict:
    print("lm_head.weight not found. Using model.embed_tokens.weight as replacement.")
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

model.load_state_dict(state_dict)  # Then load your fine-tuned weights
print('Done with model loading')
model.save_pretrained(output_dir)


# dp - Gemma
# from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
# # do the training and checkpoint saving
# checkpoint_dir = "Gemma3-1B-IT-SFT-DP-EP8"
# tag = "global_step1718"
# output_path =  checkpoint_dir
# state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag=tag) # already on cpu
# model = model.cpu() # move to cpu
# model.load_state_dict(state_dict)
# print('Done with model loading')
# model.save_pretrained(output_path)
