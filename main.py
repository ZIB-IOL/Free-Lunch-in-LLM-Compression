import getpass
import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager

import torch
import wandb

from runner import Runner
from utilities import Utils

TMP_DIR_ROOT = '/scratch/local/'

debug = "--debug" in sys.argv

defaults = dict(
    seed=0,

    #model='Qwen/Qwen2.5-72B-Instruct',
    #model='Qwen/Qwen2.5-32B-Instruct',
    #model='Qwen/Qwen2.5-14B-Instruct',
    #model='Qwen/Qwen2.5-7B-Instruct',
    #model='Qwen/Qwen2.5-3B-Instruct',
    #model='Qwen/Qwen2.5-0.5B-Instruct',
    #model='meta-llama/Llama-3.3-70B-Instruct',
    #model='meta-llama/Llama-3.1-8B-Instruct',
    #model='meta-llama/Llama-3.2-1B-Instruct',
    #model='meta-llama/Llama-2-70b-hf',
    #model='meta-llama/Llama-2-13b-hf',
    #model='meta-llama/Llama-2-7b-hf',
    #model='facebook/opt-66b',
    #model='facebook/opt-30b',
    #model='facebook/opt-13b',
    #model='facebook/opt-6.7b',
    #model='facebook/opt-2.7b',
    #model='facebook/opt-1.3b',
    model='facebook/opt-125m',
    calibration_dataset="c4",
    goal_sparsity=0.5,
    prune_method='wanda',
    sparsity_type='unstructured',    # Must be in [None, 'unstructured', '2:4', '4:8'], defaults to 'unstructured'
    ria_alpha=0.5,
    flap_metric="WIFV",
    flap_pruning_ratio=0.2,
    flap_unstr=False,
    prune_whole_matrix=False, # non-rowwise unstructured pruning

    training_mode="reconstruct", # Must be in ['retrain', 'reconstruct', None]
    peft_strategy='BlockOnlyFullFT',  # Must be in ['SelectivePEFT', 'FullFT', 'BlockOnlyFullFT']
    peft_use_bias=True,
    peft_use_ln=True,
    peft_use_lm_head=False,
    peft_use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.,
    lora_type='masked_efficient', # Must be in ['lora', 'lora_prune', 'hadamard', 'masked', 'masked_efficient', 'spp']
    lora_masking_freq=1,  # MaskedLoRA: enforce sparsity every n iterations

    # Reconstruction config
    reconstruct_n_samples=128,
    prune_n_samples=None, # set to int when retraining after pruning with a different amount of data
    reconstruct_with_max_information_data=False, # filter data by length
    propagate_sparse_activations_prune=True,
    propagate_sparse_activations_reconstruct=False,
    use_dense_targets=False,
    mask_pad_tokens=False, # use padding tokens for batching during pruning and reconstruction
                           # if False, use position ids and flat batching
    distribute_reconstruction_blocks=False,
    constant_layer_norm=False,

    # Training
    batch_size=2,
    block_size=1,         # -1 for attn/MLP split, -2 for every matrix separately
    pruning_block_size=1, # different block size for pruning
    n_iterations=0,       # for 0, we use all samples once (if n_epochs is not set, 0, or 1)
    n_epochs=1,           # 0, None, or False behave the same. for 1 or more, n_iterations is
                          # overridden and we iterate over the dataset n_epochs times
    initial_lr=1e-05,     # Initial learning rate for linear decay
    weight_decay=None,    # Defaults to 0
    gradient_accumulation_steps=1,   # Defaults to 1
    gradient_checkpointing=False,
    optim=None,           # Defaults to AdamW, must be in [None, adafactor, adamw_bnb_8bit, sgd]
    momentum=0.9,         # Defaults to 0.9
    lr_scheduler_type='linear', # Defaults to 'linear
    loss_fn='ce',                 # ce, mse, mse_normalized, cosine, distill_ce, distill_mse
    max_grad_norm=None,

    # Evaluation
    eval_zero_shot=False,
    do_eval=False,    # Enable evaluation throughout training

    # Other
    include_tokens_per_second=False,    # Defaults to False since it increases the overall runtime
    log_train_loss=False,
    log_grad_norm=False,
    )

if not debug:
    # Set everything to None recursively
    defaults = Utils.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
wandb.init(
    config=defaults,
    project='test-000',  # automatically changed in sweep
    entity=None,  # automatically changed in sweep
)
config = wandb.config
config = Utils.update_config_with_default(config, defaults)


@contextmanager
def tempdir():
    username = getpass.getuser()
    tmp_root = (TMP_DIR_ROOT if TMP_DIR_ROOT else os.getcwd()) + username
    tmp_path = os.path.join(tmp_root, 'tmp')
    if os.path.isdir(TMP_DIR_ROOT if TMP_DIR_ROOT else os.getcwd()) and not os.path.isdir(tmp_root):
        os.mkdir(tmp_root)
    if os.path.isdir(tmp_root):
        if not os.path.isdir(tmp_path): os.mkdir(tmp_path)
        path = tempfile.mkdtemp(dir=tmp_path)
    else:
        assert 'htc-' not in os.uname().nodename, "Not allowed to write to /tmp on htc- machines."
        path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
            sys.stdout.write(f"Removed temporary directory {path}.\n")
        except IOError:
            sys.stderr.write('Failed to clean up temp dir {}'.format(path))


with tempdir() as tmp_dir:
    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)
