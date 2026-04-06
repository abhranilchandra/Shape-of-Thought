import os
import random
import sys
import numpy as np
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Tuple

import torch
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, SFTConfig
from datasets import Dataset
from tasks import math  



@dataclass
class ScriptArguments:
    """
    Arguments for the full finetuning script
    """
    tokenizer_name: Optional[str] = field(
        default="Qwen/Qwen2.5-1.5B",
        metadata={"help": "Tokenizer path"},
    )
    model_name: Optional[str] = field(
        default="Qwen/Qwen2.5-1.5B",
        metadata={"help": "Model path"},
    )
    dataset_type: Optional[str] = field(
        default="math", 
        metadata={"help": "Dataset type"}
    )
    train_dataset_path: Optional[str] = field(
        default="",
        metadata={"help": "Training dataset"},
    )    
    valid_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional separate validation dataset. If None, will sample from train dataset."},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "Output directory"},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, 
        metadata={"help": "Train batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=12, 
        metadata={"help": "Eval batch size"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, 
        metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: Optional[float] = field(
        default=1e-6, 
        metadata={"help": "Learning rate"}
    )
    num_train_epochs: Optional[float] = field(
        default=10, 
        metadata={"help": "Number of epochs"}
    )
    logging_steps: Optional[int] = field(
        default=1, 
        metadata={"help": "Logging steps"}
    )
    save_steps: Optional[int] = field(
        default=10, 
        metadata={"help": "Save checkpoint steps"}
    )
    eval_steps: Optional[int] = field(
        default=1000, 
        metadata={"help": "Evaluation steps"}
    )
    seq_length: Optional[int] = field(
        default=1024, 
        metadata={"help": "Sequence length"}
    )
    warmup_steps: Optional[int] = field(
        default=100, 
        metadata={"help": "Warmup steps"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.1, 
        metadata={"help": "Warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.05, 
        metadata={"help": "Weight decay"}
    )
    optimizer_type: Optional[str] = field(
        default="adamw_torch", 
        metadata={"help": "Optimizer"}
    )
    use_validation_split: Optional[bool] = field(
        default=False, 
        metadata={"help": "Whether to use a validation split. If False, all data will be used for training."}
    )
    validation_size: Optional[int] = field(
        default=100, 
        metadata={"help": "Size of validation set to sample from training data if no separate validation set is provided."}
    )
    random_seed: Optional[int] = field(
        default=42, 
        metadata={"help": "Random seed for reproducibility"}
    )
    eval_before_train: Optional[bool] = field(
        default=False, 
        metadata={"help": "Evaluate before training"}
    )
    load_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to load checkpoint from for continued training"}
    )
    run_name: Optional[str] = field(
        default="Full_FT_Syn_Data", 
        metadata={"help": "Run name for wandb"}
    )




def save_hyperparameters(args, output_dir):
    """
    Save all hyperparameters to a log file in the output directory for reproducibility.
    
    Args:
        args: The script arguments object containing all hyperparameters
        output_dir: Directory to save the hyperparameters log
    """
    import os
    import json
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert args to dictionary
    config_dict = vars(args)
    
    # Add timestamp
    config_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Add system info
    try:
        import torch
        import platform
        import psutil
        
        config_dict['system_info'] = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else "N/A",
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_models': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            'cpu_count': psutil.cpu_count(logical=True),
            'memory_available': f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
        }
    except ImportError:
        config_dict['system_info'] = {
            'note': 'System info collection failed due to missing imports'
        }
    
    # Save as JSON
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4, sort_keys=True)
    
    # Also save as a human-readable text file
    txt_path = os.path.join(output_dir, 'training_config.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Training Configuration\n")
        f.write(f"=====================\n")
        f.write(f"Timestamp: {config_dict['timestamp']}\n\n")
        
        f.write(f"Hyperparameters\n")
        f.write(f"--------------\n")
        for key, value in sorted(config_dict.items()):
            if key not in ['timestamp', 'system_info']:
                f.write(f"{key}: {value}\n")
        
        if 'system_info' in config_dict:
            f.write(f"\nSystem Information\n")
            f.write(f"-----------------\n")
            for key, value in config_dict['system_info'].items():
                f.write(f"{key}: {value}\n")
    
    print(f"Hyperparameters saved to {config_path} and {txt_path}")
    return config_path





def set_seed(seed: int):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )




def create_datasets(args) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Create training and optional validation datasets
    
    Args:
        args: Script arguments
        
    Returns:
        Tuple of (train_dataset, valid_dataset)
        valid_dataset will be None if use_validation_split is False
    """
    print(f"Loading training dataset: {args.train_dataset_path}")
    train_dataset = Dataset.from_json(args.train_dataset_path)
    
    # Set seed for reproducibility
    set_seed(args.random_seed)
    
    # Initialize validation dataset as None
    valid_dataset = None
    
    if args.use_validation_split:
        if args.valid_dataset_path:
            # Use separate validation dataset if provided
            print(f"Loading validation dataset: {args.valid_dataset_path}")
            valid_dataset = Dataset.from_json(args.valid_dataset_path)
        else:
            # Otherwise sample from training dataset
            print(f"Sampling {args.validation_size} examples from training data for validation")
            train_dataset = train_dataset.shuffle(seed=args.random_seed)
            valid_dataset = train_dataset.select(range(args.validation_size))
            train_dataset = train_dataset.select(range(args.validation_size, len(train_dataset)))
            
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(valid_dataset)}")
    else:
        print(f"Using all data for training. Train dataset size: {len(train_dataset)}")
    
    return train_dataset, valid_dataset


# Parse arguments
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# Save hyperparameters for reproducibility
save_hyperparameters(script_args, script_args.output_dir)


# Set seed for reproducibility
set_seed(script_args.random_seed)

print("Initializing accelerator...")
#device_map = {"": Accelerator().local_process_index}

device_map= "auto"




print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, trust_remote_code=True)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set padding token to EOS token: {tokenizer.pad_token}")

print("Loading datasets...")
train_dataset, valid_dataset = create_datasets(script_args)

# Configure training arguments
training_args = SFTConfig(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_seq_length=script_args.seq_length,
    num_train_epochs=script_args.num_train_epochs,
    save_steps=script_args.save_steps,
    warmup_ratio=script_args.warmup_ratio,
    #warmup_steps=script_args.warmup_steps,
    weight_decay=script_args.weight_decay,
    optim=script_args.optimizer_type,
    bf16=True,  # Using BF16 precision
    remove_unused_columns=True,
    report_to="wandb",
    run_name=script_args.run_name,
    seed=script_args.random_seed,
)

# Configure evaluation if validation set exists
if valid_dataset is not None:
    training_args.per_device_eval_batch_size = script_args.per_device_eval_batch_size
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = script_args.eval_steps

print("Preparing data collator and formatter...")
formatting_func = partial(math.format_task, tokenizer=tokenizer)
response_template = math.get_math_response_tokens()
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[1:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# Load base model
print(f"Loading base model: {script_args.model_name}")
if script_args.load_from_checkpoint:
    print(f"Loading model from checkpoint: {script_args.load_from_checkpoint}")
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.load_from_checkpoint,
        #device_map=device_map,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        #device_map=device_map,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )

print_trainable_parameters(base_model)

# Print trainable parameters (full finetuning)
print("Trainable parameters in the base model:")
trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {trainable_params}")

# Create SFT Trainer
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    formatting_func=formatting_func,
    args=training_args,
)

# Start training
print("Starting training...")
trainer.train()

# Save final model
print("Saving final model...")
trainer.save_model(script_args.output_dir)
base_model.save_pretrained(os.path.join(script_args.output_dir, "final_checkpoint"))

print("Training complete!")
