from collections import namedtuple
import os
import sys
import random
import wandb
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import peft_methods
from utilities import Utils, check_reconstruction_error_per_matrix, check_local_reconstruction_error_per_matrix

from transformers import TrainingArguments, Trainer
import datasets

from caching_dummy import Caching
CACHE_BASE = os.path.join(os.getcwd(), 'llm_cache')

class Runner:
    def __init__(self, config, tmp_dir, debug, sweep_id):
        self.config = config
        self.tmp_dir = tmp_dir
        self.debug = debug
        self.sweep_id = sweep_id
        sys.stdout.write(f"Using temporary directory {self.tmp_dir}.\n")

        self.train_dataset_name = self.config.calibration_dataset or 'c4'

        self.token = os.environ['HF_TOKEN']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.is_opt = 'facebook/opt' in self.config.model
        self.is_llama = 'meta-llama/Llama-2' in self.config.model or 'meta-llama/Llama-3.' in self.config.model
        self.is_mistral = 'mistralai' in self.config.model
        self.is_qwen = 'Qwen' in self.config.model
        assert self.is_opt or self.is_llama or self.is_mistral or self.is_qwen, f"Model family not supported for model {self.config.model}."

        self.directoryDict = {
                'output': os.path.join(self.tmp_dir, 'output'),
            }

        self.cache_base = CACHE_BASE
            
        for dir_name in ['pretrained_models', 'datasets', 'tokenized_datasets']:
            dir_path = os.path.join(self.cache_base, dir_name)
            self.directoryDict[dir_name] = dir_path
            os.makedirs(dir_path, exist_ok=True)
        os.makedirs(self.directoryDict['output'], exist_ok=True)


    def get_llm(self, model_name, sweep_id=None):
        torch_dtype = torch.float16 # In the original setup, this was specified as torch.float16

        device_map = "auto"
        if self.config.distribute_reconstruction_blocks:
            sys.stdout.write(f"Distributing reconstruction submodels across {torch.cuda.device_count()} GPUs.")
            #assert hasattr(self.model, "hf_device_map"), "model.hf_device_map must be defined."
            num_gpus = torch.cuda.device_count()
            model_name = self.config.model
            assert model_name.startswith("meta-llama/Llama-2") or model_name.startswith("meta-llama/Llama-3")\
                or model_name.startswith("Qwen/Qwen2") or model_name.startswith("facebook/opt"),\
                f"Model {model_name} not supported for distributing reconstruction blocks."
            if self.config.model.startswith("facebook/opt"):
                device_map = {
                    "model.embed_tokens": 0,
                    "model.norm": num_gpus - 1,
                    "lm_head": num_gpus - 1,
                }
                n_blocks_map = {"125m": 12, "1.3b": 24, "6.7b": 32}
            else:
                device_map = {
                    "model.embed_tokens": 0,
                    "model.norm": num_gpus - 1,
                    "model.rotary_emb": 0,
                    "lm_head": num_gpus - 1,
                }
                n_blocks_map = {"13b": 40, "7B": 28, "8B": 32, "32B": 64, "70B": 80, "72B": 80}
            
            # evenly distribure the layers of each reconstruction block across the GPUs
            gpu_map = torch.floor(torch.linspace(0, num_gpus, max(self.config.block_size, 1)+1))[:-1].int()
            # number of transformer blocks per model
            num_params = self.config.model.split("-")[-2]
            if not num_params in n_blocks_map:
                num_params = self.config.model.split("-")[-1]
            assert num_params in n_blocks_map, f"Model {self.config.model} not in :n_blocks_map: dict."
            for i in range(n_blocks_map[num_params]):
                device_map[f"model.layers.{i}"] = gpu_map[i % max(self.config.block_size, 1)].item()

        if sweep_id is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                cache_dir=self.directoryDict['pretrained_models'],
                low_cpu_mem_usage=True,
                device_map=device_map,
                attn_implementation=self.config.attn_implementation or "flash_attention_2",
                quantization_config=None,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(self.config.checkpointdir, sweep_id, "best_model"),
                local_files_only=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map=device_map,
                attn_implementation=self.config.attn_implementation or "flash_attention_2",
                quantization_config=None,
            )
            
        if model.config.max_position_embeddings > 4096:
            model.seqlen = 4096
            sys.stdout.write(f"Avoiding OOM by setting model.seqlen to 4096 for {model_name}.\n")
        else:
            model.seqlen = model.config.max_position_embeddings
        return model

    def get_trainer(self, tokenized_datasets, batch_size=None, seed=None):

        max_steps = 1

        # Huggingface trainer approach

        train_args = {
            "seed": seed if seed is not None else self.config.seed,
            # Training hyperparameters
            "per_device_train_batch_size": batch_size or self.config.batch_size,
            "per_device_eval_batch_size": batch_size or self.config.batch_size,
            "max_steps": max_steps,
            "learning_rate": 0.1,
            "lr_scheduler_type": 'linear',  # Linear learning rate decay
            "warmup_ratio": 0.1,  # Warmup ratio for linear learning rate scheduler, keep fixed at 10%
            "weight_decay": 0.,  # Strength of weight decay
            "max_grad_norm": 1.0,
            
            # Evaluation
            "evaluation_strategy": 'no',
            "eval_steps": 100,            

            # Additional optimization parameters
            "gradient_accumulation_steps": 1,  # Number of updates steps to accumulate before performing a backward/update pass.
            "fp16": True,  # Use mixed precision
            "gradient_checkpointing": False,  # If true, enables gradient checkpointing to save memory
            "optim": 'adamw_torch',  # Use adamw_torch, adafactor or adamw_bnb_8bit

            # Logging
            "report_to": "wandb",  # Enable logging to W&B
            "logging_steps": 100,  # Log every X updates steps
            "logging_first_step": True,    # Log also the first step
            
            # Model Checkpointing
            "output_dir": self.directoryDict['output'],
            #"overwrite_output_dir": True,
            "save_strategy": "no", # Do not save the model checkpoints
        }
        try:
            training_args = TrainingArguments(**train_args)
        except: # newer transformers versions
            train_args["eval_strategy"] = train_args.pop("evaluation_strategy")
            training_args = TrainingArguments(**train_args)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            #tokenizer=self.tokenizer,
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        return trainer

    def change_model_state(self, model, train: bool):
        sys.stdout.write(f"Changing model state to {'train' if train else 'eval'} mode.\n")
        if train:
            model.train()
            model.enable_input_require_grads() # Needed for PEFT: https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
        else:
            model.eval()


    def get_dataset(self, dataset_name: str) -> tuple:
        """
        Returns the tokenized datasets for the given dataset name and uses the caching module to cache the tokenized dataset.
        """
        sys.stdout.write(f"Loading {dataset_name}.\n")
        assert dataset_name in ['wikitext2', 'c4', 'minipile'], f"Dataset {dataset_name} not supported."
        data_path = Caching.get_dataset_root(dataset_name, tokenizer=self.tokenizer, seqlen=self.model.seqlen, cache_base=self.cache_base)
                
        if dataset_name == 'wikitext2':
            tmp = {'input_ids': torch.load(os.path.join(data_path, 'input_ids.pt'), weights_only=True),
                'attention_mask': torch.load(os.path.join(data_path, 'attention_mask.pt'), weights_only=True)}
            tokenized_datasets = transformers.tokenization_utils_base.BatchEncoding(tmp)
        else:
            tokenized_datasets = datasets.load_from_disk(data_path)

        if dataset_name in ['c4', 'minipile']:
            # Take only 100 random samples for validation
            tokenized_datasets['validation'] = tokenized_datasets['validation'].shuffle(seed=self.config.seed).select(range(100))

        return tokenized_datasets
    
            
    def make_model_param_efficient(self):
        sys.stdout.write(f"Percentage of parameters with grad without PEFT: {Utils.get_percentage_of_trainable_parameters(self.model)}\n")

        # Enable grad for all parameters that correspond to the peft strategy at stake
        assert hasattr(peft_methods, self.config.peft_strategy), f"PEFT strategy {self.config.peft_strategy} not implemented."
        self.peft_strategy = getattr(peft_methods, self.config.peft_strategy)(model=self.model, runner=self, config=self.config, total_iterations=self.n_iterations, is_reconstruct=False)
        self.peft_strategy.select_peft_layers()

        for param in self.model.parameters():
            if param.requires_grad:
                # Important: Set trainable parameters to float32, otherwise this won't work with fp16=True -> https://github.com/huggingface/peft/issues/341#issuecomment-1519460307
                param.data = param.data.float()
        
        sys.stdout.write(f"Percentage of parameters with grad with PEFT: {Utils.get_percentage_of_trainable_parameters(self.model)}\n")


    def run(self):
        # Setting seeds for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.random.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        sys.stdout.write(f"Running on node {self.config.computer} with seed {self.config.seed}.\n")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'], use_fast=False)
        
        sweep_ids = self.config.sweep_ids.split(",") + [None]

        for i, sweep_id1 in enumerate(sweep_ids if not self.config.check_local_errors else [None]):
            for j, sweep_id2 in enumerate(sweep_ids[i+1:] if not self.config.check_local_errors else sweep_ids[:-1]):
                model1 = self.get_llm(self.config.model, sweep_id1)
                model2 = self.get_llm(self.config.model, sweep_id2)
                self.model = model1
                if self.is_llama or self.is_mistral:
                    #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.tokenizer.pad_token = self.tokenizer.eos_token # For shorter sequences, the eos_token is used as padding token
                    model1.resize_token_embeddings(len(self.tokenizer))
                    model2.resize_token_embeddings(len(self.tokenizer))

                # Reconfigure the device in the case of multiple GPUs (set to the device of lm_head)
                if torch.cuda.device_count() > 1:
                    device = model1.hf_device_map["lm_head"]
                    sys.stdout.write(f"Using {torch.cuda.device_count()} GPUs - setting self.device = {device}.\n")
                args = namedtuple('args', ['device', 'block_size', 'reconstruct_n_samples', 'seed', 'batch_size',
                                            'train_dataset_name', 'mask_pad_tokens', 'reconstruct_with_max_information_data'])(
                    self.device, 1, self.config.reconstruct_n_samples, self.config.seed, 1,
                    self.train_dataset_name, False, self.config.reconstruct_with_max_information_data
                )
                if self.config.check_local_errors:
                    error_train_mean, error_val_mean, error_train_max, error_val_max = check_local_reconstruction_error_per_matrix(model1, model2, args, self)
                else:
                    error_train_mean, error_val_mean, error_train_max, error_val_max = check_reconstruction_error_per_matrix(model1, model2, args, self)
                for key, val in error_train_mean.items():
                    wandb.run.summary[str(sweep_id1) + "_" + str(sweep_id2) + "_" + "mean_" + key] = val
                wandb.log(error_train_mean, commit=True)
                for key, val in error_val_mean.items():
                    wandb.run.summary[str(sweep_id1) + "_" + str(sweep_id2) + "_" + "mean_" + key] = val
                wandb.log(error_val_mean, commit=True)
                for key, val in error_train_max.items():
                    wandb.run.summary[str(sweep_id1) + "_" + str(sweep_id2) + "_" + "max_" + key] = val
                wandb.log(error_train_max, commit=True)
                for key, val in error_val_max.items():
                    wandb.run.summary[str(sweep_id1) + "_" + str(sweep_id2) + "_" + "max_" + key] = val
                wandb.log(error_val_max, commit=True)