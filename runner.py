from collections import namedtuple
import getpass
import os
import sys
import time
import random
import wandb
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import peft_methods
from utilities import Utils

from transformers import TrainingArguments, Trainer
import datasets
import platform
from prune_methods import PruneMethod
from prune_flap import PruneFLAP

from caching_dummy import Caching

class Runner:
    def __init__(self, config, tmp_dir, debug):
        self.config = config
        self.tmp_dir = tmp_dir
        self.debug = debug
        sys.stdout.write(f"Using temporary directory {self.tmp_dir}.\n")

        self.train_dataset_name = self.config.calibration_dataset or 'c4'

        self.token = os.environ['HF_TOKEN']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.is_opt = 'facebook/opt' in self.config.model
        self.is_llama = 'meta-llama/Llama-2' in self.config.model or 'meta-llama/Llama-3.' in self.config.model
        self.is_mistral = 'mistralai' in self.config.model
        self.is_qwen = 'Qwen' in self.config.model
        assert self.is_opt or self.is_llama or self.is_mistral or self.is_qwen, f"Model family not supported for model {self.config.model}."

        # If n_iterations is 0, we set it so every sample is used once (at least when batch_size * gradient_accumulation_steps divides the number of samples)
        # If n_epochs is set, we iterate over the dataset n_epochs times and ignore n_iterations
        if self.config.n_iterations == 0 or self.config.n_epochs:
            self.n_iterations = self.config.reconstruct_n_samples
            self.n_iterations = int(np.ceil(self.n_iterations / self.config.batch_size))
            if self.config.gradient_accumulation_steps is not None:
                self.n_iterations = self.n_iterations // self.config.gradient_accumulation_steps\
                                    + int(self.n_iterations % self.config.gradient_accumulation_steps > 0)
            if self.config.n_epochs:
                assert self.config.n_epochs > 0, "n_epochs must be greater than 0."
                self.n_iterations = self.config.n_epochs * self.n_iterations
        else:
            self.n_iterations = self.config.n_iterations

        self.cache_base = os.path.join(os.getcwd(), 'llm_cache')
        self.directoryDict = {
                'datasets_tokenized_permanent': '/software/ais2t/datasets/huggingface_tokenized',   # Directory for permanent tok. datasets on z1
                'output': os.path.join(self.tmp_dir, 'output'),  # Directory for model checkpoints, which we redirect to tmp, so they get deleted
            }
            
        for dir_name in ['pretrained_models', 'datasets', 'tokenized_datasets']:
            dir_path = os.path.join(self.cache_base, dir_name)
            self.directoryDict[dir_name] = dir_path
            os.makedirs(dir_path, exist_ok=True)
        os.makedirs(self.directoryDict['output'], exist_ok=True)

        self.check_config(self.config)

        # Variables to be defined
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.peft_strategy = None

        # Other metrics
        self.time_for_pruning, self.time_for_retraining = None, None


    def check_config(self, config):
        args_without_default = ['model', 'goal_sparsity', 'prune_method', 'batch_size', 'reconstruct_n_samples']
        for arg in args_without_default:
            assert getattr(config, arg, None) is not None, f"Argument {arg} must be specified."
        if config.training_mode == "reconstruct":
            assert config.block_size is not None, "block_size must be specified for training_mode == 'reconstruct'."


    def get_llm(self, model_name):
        torch_dtype = torch.float16 # In the original setup, this was specified as torch.float16

        device_map = "auto"
        if self.config.distribute_reconstruction_blocks:
            sys.stdout.write(f"Distributing reconstruction submodels across {torch.cuda.device_count()} GPUs.")
            assert hasattr(self.model, "hf_device_map"), "model.hf_device_map must be defined."
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

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=self.directoryDict['pretrained_models'],
            low_cpu_mem_usage=True,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            quantization_config=None,
        )

        if model.config.max_position_embeddings > 4096:
            model.seqlen = 4096
            sys.stdout.write(f"Avoiding OOM by setting model.seqlen to 4096 for {model_name}.\n")
        else:
            model.seqlen = model.config.max_position_embeddings
        return model


    def change_model_state(self, train: bool):
        sys.stdout.write(f"Changing model state to {'train' if train else 'eval'} mode.\n")
        if train:
            self.model.train()
            self.model.enable_input_require_grads() # Needed for PEFT: https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
        else:
            self.model.eval()


    def eval_on_wikitext(self):
        train_state = self.model.training
        self.change_model_state(train=False)

        # Load test dataset
        testLoader = self.get_dataset('wikitext2')

        sys.stdout.write(f"Evaluating wikitext2.\n")
        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            ppl_test = Utils.compute_ppl_wikitext(self.model, testLoader, 1, self.device)
        self.change_model_state(train=train_state)
        return ppl_test


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


    def log_metrics(self, state, additional_metrics=None):
        metrics = {
            'metrics/sparsity': Utils.check_sparsity(self.model),
            'metrics/ppl_test': self.eval_on_wikitext(),
        }

            
        if additional_metrics is not None:
            keys_to_remove = []
            for key, val in additional_metrics.items():
                if "train_loss" in key:
                    loss_table = wandb.Table(columns=["train_loss"], data=[[v] for v in val])
                    wandb.log({
                        f"metrics/{key}": loss_table
                    })
                    keys_to_remove.append(key)
                if "val_loss" in key:
                    loss_table = wandb.Table(columns=["val_loss"], data=[[v] for v in val])
                    wandb.log({
                        f"metrics/{key}": loss_table
                    })
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                additional_metrics.pop(key)

            metrics.update(additional_metrics)
            

        for time_metric in ['time_for_pruning', 'time_for_retraining']:
            if getattr(self, time_metric) is not None:
                metrics[f'metrics/{time_metric}'] = getattr(self, time_metric)
                
                # Set to None to avoid logging the same metric twice
                setattr(self, time_metric, None)

        # Log the metrics to the wandb summary with the given state as prefix
        for key, val in metrics.items():
            wandb.run.summary[f'{state}/{key}'] = val


        step = self.trainer.state.global_step if self.trainer is not None else 0
        commit = self.trainer is not None
        sys.stdout.write(f"Logging metrics to wandb with step {step} and commit {commit}.\n")
        wandb.log(metrics, step=step, commit=commit)
        sys.stdout.write(f"Finished logging metrics to wandb with step {step} and commit {commit}.\n")

    def train_on_c4(self):
        #assert self.n_iterations % 100 == 0, "Currently only supports n_iterations that are a multiple of 100."
        if self.n_iterations % 100 != 0:
            sys.stdout.write(f"Warning: n_iterations not a multiple of 100, last logging steps might be omitted.\n")

        self.change_model_state(train=True)

        # Load the tokenized dataset
        tokenized_datasets = self.get_dataset('c4')
        if self.config.reconstruct_n_samples is not None:
            tokenized_datasets['train'] = tokenized_datasets['train'].shuffle(seed=self.config.seed).select(range(self.config.reconstruct_n_samples))


        self.trainer = self.get_trainer(tokenized_datasets)

        self.peft_strategy.iteration_getter = lambda: self.trainer.state.global_step
        self.trainer.train()


    def get_trainer(self, tokenized_datasets, batch_size=None, seed=None):

        max_steps = self.n_iterations
        if self.config.training_mode == None: # for unfiltered data when only pruning is done
            max_steps = 1

        # Huggingface trainer approach

        train_args = {
            "seed": seed if seed is not None else self.config.seed,
            # Training hyperparameters
            "per_device_train_batch_size": batch_size or self.config.batch_size,
            "per_device_eval_batch_size": batch_size or self.config.batch_size,
            "max_steps": max_steps,
            "learning_rate": float(self.config.initial_lr),
            "lr_scheduler_type": self.config.lr_scheduler_type or 'linear',  # Linear learning rate decay
            "warmup_ratio": 0.1,  # Warmup ratio for linear learning rate scheduler, keep fixed at 10%
            "weight_decay": float(self.config.weight_decay) if self.config.weight_decay is not None else 0.,  # Strength of weight decay
            "max_grad_norm": float(self.config.max_grad_norm) if self.config.max_grad_norm is not None else 1.0,
            
            # Evaluation
            "evaluation_strategy": 'steps' if self.config.do_eval else 'no',
            "eval_steps": 100,            

            # Additional optimization parameters
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps or 1,  # Number of updates steps to accumulate before performing a backward/update pass.
            "fp16": True,  # Use mixed precision
            "gradient_checkpointing": self.config.gradient_checkpointing,  # If true, enables gradient checkpointing to save memory
            "optim": self.config.optim or 'adamw_torch',  # Use adamw_torch, adafactor or adamw_bnb_8bit

            # Logging
            "report_to": "wandb",  # Enable logging to W&B
            "logging_steps": 100,  # Log every X updates steps
            "logging_first_step": True,    # Log also the first step
            "include_tokens_per_second": self.config.include_tokens_per_second or False,   # Log the tokens per second, however this increases the overall runtime

            # Model Checkpointing
            "output_dir": self.directoryDict['output'],
            "overwrite_output_dir": True,
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
            tokenizer=self.tokenizer,
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        return trainer


    def prune_model(self, sparsity):
        """Prune the model using the specified method."""
        # Check whether we retrain and no peft method is selected, then we have to keep the masks
        keep_masks = self.config.peft_strategy in ['FullFT', 'BlockOnlyFullFT'] and self.config.training_mode == 'retrain'
        sys.stdout.write(f"Keeping masks: {keep_masks}.\n")

        # Define the necessary args
        args = {
            'training_mode': self.config.training_mode,
            'reconstruct': self.config.training_mode == 'reconstruct',
            'reconstruct_n_samples': self.config.reconstruct_n_samples if self.config.prune_n_samples is None else self.config.prune_n_samples,
            'n_iterations': self.n_iterations,
            'batch_size': self.config.batch_size,
            'block_size': self.config.block_size,
            'pruning_block_size': self.config.pruning_block_size,
            'initial_lr': float(self.config.initial_lr),
            'seed': self.config.seed,
            'sparsity_ratio': sparsity,
            'cache_dir': self.directoryDict['datasets'],
            'tokenizer': self.tokenizer,
            'device': self.device,
            'keep_masks': keep_masks,
            'propagate_sparse_activations_prune': self.config.propagate_sparse_activations_prune,
            'propagate_sparse_activations_reconstruct': self.config.propagate_sparse_activations_reconstruct,
            'ria_alpha': self.config.ria_alpha,
            'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
            'reconstruct_with_max_information_data': self.config.reconstruct_with_max_information_data,
            'prune_whole_matrix': self.config.prune_whole_matrix,
            'use_dense_targets': self.config.use_dense_targets,
            'loss_fn': self.config.loss_fn,
            'train_dataset_name': self.train_dataset_name,
            'mask_pad_tokens': self.config.mask_pad_tokens,
            'constant_layer_norm': self.config.constant_layer_norm,
            'log_train_loss': self.config.log_train_loss,
            'log_grad_norm': self.config.log_grad_norm,
            'optim': self.config.optim,
            'momentum': self.config.momentum,
        }

        # Handle n:m sparsity
        prune_n, prune_m = 0, 0
        sparsity_type = self.config.sparsity_type or 'unstructured'
        assert sparsity_type in ['unstructured', '2:4', '4:8'], f"Sparsity type {sparsity_type} not supported."
        if sparsity_type != 'unstructured':
            assert self.config.goal_sparsity == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity. Note: currently, this implemented in the wrong way, i.e. pruning n out of m, instead of m-n out of m."
            prune_n, prune_m = map(int, sparsity_type.split(":"))

        # Define a named tuple to pass the args (since then they can be accessed as attributes)
        NamedTupleClass = namedtuple('ArgTuple', args)
        args_ = NamedTupleClass(**args)
        if self.config.prune_method == 'flap':
            pruneMethod = PruneFLAP(runner=self, args=args_, prune_method=self.config.prune_method, model=self.model, prune_n=prune_n, prune_m=prune_m)
        else:
            pruneMethod = PruneMethod(runner=self, args=args_, prune_method=self.config.prune_method, model=self.model, prune_n=prune_n, prune_m=prune_m)
        pruneMethod.prune()
        self.pruneMethod = pruneMethod

        if self.config.prune_method == 'flap' and self.config.training_mode == 'reconstruct':
            args["reconstruct"] = True
            args["training_mode"] = "reconstruct"
            args["reconstruct_n_samples"] = self.config.reconstruct_n_samples
            NamedTupleClass = namedtuple('ArgTuple', args)
            pruneMethod = PruneMethod(runner=self, args=NamedTupleClass(**args), prune_method=self.config.prune_method, model=self.model, prune_n=prune_n, prune_m=prune_m)
            pruneMethod.prune(do_prune=False)
            self.pruneMethod = pruneMethod

        if self.config.training_mode == 'reconstruct':
            previous_train_state = self.model.training
            self.change_model_state(train=True)
            self.change_model_state(train=previous_train_state)
        

    def get_zeroshot_metrics(self, debug=False):
        # Requires a custom module, as outlined here https://github.com/locuslab/wanda/tree/main
        sys.stdout.write(f"Evaluating zero-shot performance.\n")
        train_state = self.model.training
        self.change_model_state(train=False)
        task_list = ["boolq", "rte", "hellaswag","winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        if debug:
            task_list = task_list[:1]
        results = Utils.eval_zero_shot(self.config.model, self.model, task_list)['results']
        zero_shot_acc = {f"metrics/zero_shot_acc_{task}": results[task]['acc'] for task in task_list}
        zero_shot_stderr = {f"metrics/zero_shot_accstderr_{task}": results[task]['acc_stderr'] for task in task_list}
        avg_zero_shot_acc = np.mean([results[task]['acc'] for task in task_list])
        zero_shot_metrics = {
            'metrics/avg_zero_shot_acc': avg_zero_shot_acc,
            **zero_shot_acc,
            **zero_shot_stderr,
        }
        self.change_model_state(train=train_state)
        return zero_shot_metrics


    def run(self):
        # Setting seeds for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.random.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        sys.stdout.write(f"Running on node {self.config.computer} with seed {self.config.seed}.\n")

        sys.stdout.write(f"Loading LLM model {self.config['model']}.\n")
        # Load model and tokenizer
        self.model = self.get_llm(self.config['model'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'], use_fast=False)

        if self.is_llama or self.is_mistral:
            #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = self.tokenizer.eos_token # For shorter sequences, the eos_token is used as padding token
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Reconfigure the device in the case of multiple GPUs (set to the device of lm_head)
        if torch.cuda.device_count() > 1:
            self.device = self.model.hf_device_map["lm_head"]
            sys.stdout.write(f"Using {torch.cuda.device_count()} GPUs - setting self.device = {self.device}.\n")
            
        # Prune the model
        additional_metrics = None
        if self.config.goal_sparsity > 0:
            t_start = time.time()
            self.prune_model(self.config['goal_sparsity'])  
            self.time_for_pruning = time.time() - t_start
            if self.config.log_train_loss:
                if additional_metrics is None:
                    additional_metrics = {}
                for key, val in self.pruneMethod.train_losses.items():
                    additional_metrics[key] = val
                if self.pruneMethod.val_losses is not None:
                    for key, val in self.pruneMethod.val_losses.items():
                        additional_metrics[key] = val

            if self.config.log_grad_norm:
                if additional_metrics is None:
                    additional_metrics = {}
                for key, val in self.pruneMethod.grad_norms.items():
                    additional_metrics[key] = val

        self.log_metrics(state='pruned', additional_metrics=additional_metrics)
        additional_metrics = None
        if self.config.training_mode == 'retrain':
            # Make the model parameter efficient
            self.make_model_param_efficient()

            # Fine-tune the model on C4
            t_start = time.time()
            self.train_on_c4()
            self.time_for_retraining = time.time() - t_start

            # Potentially merge the adapters, but first, collect additional metrics that are only available before merging
            additional_metrics = {'metrics/fraction_of_trainable_parameters': Utils.get_percentage_of_trainable_parameters(self.model)}
            self.peft_strategy.at_train_end()

        if self.config.training_mode == 'retrain' or self.config.training_mode == 'reconstruct':
            self.log_metrics(state='retrained', additional_metrics=additional_metrics)

        # Evaluate zero-shot performance
        if self.config.eval_zero_shot:
            zero_shot_metrics = self.get_zeroshot_metrics(debug=self.debug)

            # Log the metrics to the wandb summary
            for key, val in zero_shot_metrics.items():
                wandb.run.summary[key] = val
            wandb.log(zero_shot_metrics, commit=True)