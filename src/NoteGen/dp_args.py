"""Compilation of all the arguments."""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import transformers

MODEL_CONFIG_CLASSES = list(transformers.MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

TRUE_TAGS = ('y', 'yes', 't', 'true')


# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from "
                    "scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    static_lm_head: str = field(default='no')
    static_embedding: str = field(default='no')
    attention_only: str = field(default="no")
    bias_only: str = field(default="no")

    def __post_init__(self):
        self.static_lm_head = self.static_lm_head.lower() in TRUE_TAGS
        self.static_embedding = self.static_embedding.lower() in TRUE_TAGS
        self.attention_only = self.attention_only.lower() in TRUE_TAGS
        self.bias_only = self.bias_only.lower() in TRUE_TAGS


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_data_file: str = field(default=None, metadata={"help": "Path to the training data file."})
    eval_data_file: Optional[str] = field(default=None, metadata={"help": "Path to the evaluation data file."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    ema_model_averaging: str = field(default="no")
    ema_model_gamma: float = field(default=0.99)
    ema_model_start_from: int = field(default=1000)
    lr_decay: str = field(default="yes")
    eval_epochs: int = field(default=999)

    deepspeed_config: str = field(default=None)
    num_GPUs: int = field(default=1)
    logical_batch_size: int = field(default=None)

    save_at_last: str = field(default="no", metadata={"help": "Save at the end of training."})

    def __post_init__(self):
        super(TrainingArguments, self).__post_init__()
        self.ema_model_averaging = (self.ema_model_averaging.lower() in ('y', 'yes'))
        self.lr_decay = (self.lr_decay.lower() in ('y', 'yes'))
        self.save_at_last = (self.save_at_last in ('y', 'yes'))


@dataclass
class PrivacyArguments:
    """Arguments for differentially private training."""
    per_example_max_grad_norm: float = field(
        default=.1, metadata={
            "help": "Clipping 2-norm of per-sample gradients."
        }
    )
    noise_multiplier: float = field(
        default=None, metadata={
            "help": "Standard deviation of noise added for privacy; if `target_epsilon` is specified, "
                    "use the one searched based budget"
        }
    )
    target_epsilon: float = field(
        default=None, metadata={
            "help": "Privacy budget; if `None` use the noise multiplier specified."
        }
    )
    target_delta: float = field(
        default=None, metadata={
            "help": "Lax probability in approximate differential privacy; if `None` use 1 / len(train_data)."
        }
    )
    accounting_mode: str = field(
        default="rdp", metadata={"help": "One of `rdp`, `glw`, `all`."}
    )
    non_private: str = field(default="no")
    clipping_mode: str = field(default="ghost")
    clipping_fn: str = field(default="automatic")
    clipping_style: str = field(default="all-layer")
    torch_seed_is_fixed: bool = field(default=True)

    def __post_init__(self):
        self.non_private = self.non_private.lower() in ('y', 'yes')
