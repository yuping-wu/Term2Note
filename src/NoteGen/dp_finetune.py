# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.

Modified to accommodate Llama.
"""
import json
import os
import logging
from tqdm import tqdm
from ml_swissknife import utils
import torch
from datasets import Dataset
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from dp_args import (
    DataTrainingArguments,
    ModelArguments,
    PrivacyArguments,
    TrainingArguments
)
from fastDP import PrivacyEngine
from fastDP import PrivacyEngine_Distributed_Stage_2_and_3
from fastDP import PrivacyEngine_Distributed_extending
from dp_trainer import CustomizedTrainer

logger = logging.getLogger(__name__)

def load_jsonl(file):
    data = []
    with open(file, "r") as f:
        for l in f:
            data.append(json.loads(l))
    return data

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, PrivacyArguments, TrainingArguments)
    )
    model_args, data_args, privacy_args, training_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    privacy_args: PrivacyArguments

    #################################
    ##### Argument Sanity Check #####
    #################################
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    
    #################################
    ######### Setup Logging #########
    #################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Debug mode
    if training_args.debug:
        import warnings
        warnings.filterwarnings("error")

    #################################
    ########## Load Model ###########
    #################################
    logger.info("Loading model from %s......", model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.tie_word_embeddings = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        # add_eos_token = True,
        # add_bos_token=False,
        use_fast=True,
    )
    if "gemma" in model_args.model_name_or_path:
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            config=config,
        )
        # raise NotImplementedError("Gemma3ForCausalLM is not supported yet.")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            config=config,
            # device_map="auto",
        )
    # Clone the embedding into the lm_head for better initialization.
    lm_head = model.get_output_embeddings()
    embedding = model.get_input_embeddings()
    lm_head.weight.data.copy_(embedding.weight.data)
    print(f'Cloning initial embedding into lm_head, '
          f'checking norms... \n'
          f'\tlm_head: {lm_head.weight.norm()}, embedding: {embedding.weight.norm()}')
    torch.testing.assert_allclose(lm_head.weight, embedding.weight)
    del lm_head, embedding
    model.requires_grad_(True)
    model.gradient_checkpointing_enable()
    logger.info("Model loaded.")


    #################################
    ######## Prepare Dataset ########
    #################################
    logger.info("Loading dataset from %s......", data_args.train_data_file)
    formatted_data = load_jsonl(data_args.train_data_file)
    def preprocess_function(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, 
            # add_generation_prompt=True
        )
        tokenized_input = tokenizer(
            text,
            truncation=True,
            max_length=8192,
            # padding=True
        )
        # tokenized_input["labels"] = tokenized_input["input_ids"].copy()
        return tokenized_input
    # apply chat format and tokenize
    # formatted_data = [{"text": tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)} for item in tqdm(loaded_data)]
    dataset_size = len(formatted_data)
    train_size = int(0.9 * dataset_size)
    # train_data = formatted_data[:train_size]
    # val_data = formatted_data[train_size:]
    # print(formatted_data[0])
    train_raw_dataset = Dataset.from_list(formatted_data[:train_size])
    val_raw_dataset = Dataset.from_list(formatted_data[train_size:])
    train_dataset = train_raw_dataset.map(
        preprocess_function,
        batched=False,
        num_proc=training_args.dataloader_num_workers,
        remove_columns=train_raw_dataset.column_names,
        load_from_cache_file=False
    )
    val_dataset = val_raw_dataset.map(
        preprocess_function,
        batched=False,
        num_proc=training_args.dataloader_num_workers,
        remove_columns=val_raw_dataset.column_names
    )
    logger.info("Dataset loaded.")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    #################################
    ############# Train #############
    #################################
    trainer = CustomizedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        # data_args=data_args,
        privacy_args=privacy_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        # generation_stuff=generation_stuff,
    )

    # Massage the parameters.
    if model_args.attention_only:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if 'c_attn.weight' in name:
                param.requires_grad_(True)
    elif model_args.bias_only:
        for name, param in model.named_parameters():
            if '.bias' not in name:
                param.requires_grad_(False)        
        if model_args.static_lm_head and hasattr(model, 'lm_head'):
            model.lm_head.requires_grad_(False)
    else:
        model.requires_grad_(True)
        if model_args.static_lm_head:
            model.get_output_embeddings().requires_grad_(False)
        if model_args.static_embedding:
            model.get_input_embeddings().requires_grad_(False)
            model.transformer.wpe.requires_grad_(False)
    print(f"bias_only: {model_args.bias_only} | attention_only: {model_args.attention_only}")

    params = tuple(param for param in model.parameters() if param.requires_grad)
    names = tuple(name for name, param in model.named_parameters() if param.requires_grad)
    num_trainable_params = sum(param.numel() for param in params)
    print(f"Number of trainable params: {num_trainable_params / 1e6:.4f} million")
    print(f'Number of total params: {sum(param.numel() for param in model.parameters()) / 1e6:.3f} million')
    print(f"Number of named params: {len(names)}")

    optimizer = torch.optim.AdamW(
        params=params,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    trainer.optimizer = optimizer

    # Create the lr_scheduler.
    try:
        num_GPUs = torch.distributed.get_world_size()
    except:
        num_GPUs = 1
    if training_args.logical_batch_size!=None:
        trainer.args.gradient_accumulation_steps=training_args.logical_batch_size/training_args.per_device_train_batch_size/num_GPUs
    else:
        training_args.logical_batch_size=trainer.args.gradient_accumulation_steps*training_args.per_device_train_batch_size*num_GPUs
    num_update_steps_per_epoch = len(trainer.get_train_dataloader()) // trainer.args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, num_update_steps_per_epoch)
    t_total = int(num_update_steps_per_epoch * training_args.num_train_epochs)
    if training_args.lr_decay:
        trainer.lr_scheduler = get_linear_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=t_total,
        )
    else:
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer,
            lr_lambda=lambda step: 1.0,
        )
    
    # Setup privcay
    if privacy_args.non_private:
        privacy_args.noise_multipler = 0.
        privacy_args.per_example_max_grad_norm = None
    else:
        # origin_params=None if model_args.bias_only or model_args.attention_only or training_args.deepspeed_config else ['wte','wpe']
        # privacy_engine = PrivacyEngine(
        # privacy_engine = PrivacyEngine_Distributed_Stage_2_and_3(
        privacy_engine = PrivacyEngine_Distributed_extending(
            module=model,
            batch_size=training_args.logical_batch_size,
            sample_size=len(train_dataset),
            epochs=training_args.num_train_epochs,
            max_grad_norm=privacy_args.per_example_max_grad_norm,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            accounting_mode=privacy_args.accounting_mode,
            # clipping_mode=privacy_args.clipping_mode,
            # clipping_fn=privacy_args.clipping_fn,
            # clipping_style=privacy_args.clipping_style,
            clipping_style='layer-wise',
            # origin_params=origin_params,
            num_GPUs=num_GPUs,
            torch_seed_is_fixed=True,
            grad_accum_steps=trainer.args.gradient_accumulation_steps,
        )

        # Originally, these could have been null.
        privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        privacy_args.target_delta = privacy_engine.target_delta

        print('privacy_args: ')
        print(json.dumps(privacy_args.__dict__, indent=4))
        # if not training_args.deepspeed_config:
        #     privacy_engine.attach(optimizer)
        # privacy_engine.attach(optimizer)
    
    # Start training
    torch.cuda.empty_cache()
    if training_args.do_train:
        all_args = {
            **training_args.__dict__,
            **data_args.__dict__,
            **model_args.__dict__,
            **privacy_args.__dict__,
        }
        utils.jdump(
            all_args,
            os.path.join(training_args.output_dir, 'argparse.json'),
            default=lambda x: str(x),
        )

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        logger.info("*** Train ***")
        logger.info(
            f"Training set size: {len(train_dataset)}, "
            f"per_device_train_batch_size: {training_args.per_device_train_batch_size}, "
            f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}"
        )
        
        trainer.train(model_path=None)
        if training_args.save_at_last:
            trainer.save_model()
            # trainer.model.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        output = trainer.evaluate(log_results=False)
        utils.jdump(
            output,
            os.path.join(training_args.output_dir, "final_results.json"),
        )

        logger.info("***** Eval results *****")
        logger.info(output)
    

    

if __name__ == "__main__":
    main()
