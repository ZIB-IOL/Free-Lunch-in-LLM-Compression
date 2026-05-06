import getpass
import os
import shutil
import socket
import sys
import tempfile
from contextlib import contextmanager
import wandb

from check_act_dist import Runner
from utilities import Utils

TMP_DIR_ROOT = 'PATH_TO_TMP_DIR'

# This script takes a list of sweep ids, loads the corresponding checkpoints and compares the distances of each intermediate activation
# both pairwise between the models and between each model and the dense model

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
    reconstruct_n_samples=32,
    batch_size=2,
    attn_implementation="flash_attention_2",
    sweep_ids="comma-separated list of sweep ids", # list of ids of sweeps which saved model checkpoints to compare

    distribute_reconstruction_blocks=False,
    reconstruct_with_max_information_data=False,
    check_local_errors=False, # check errors locally per matrix or accumulating if False
    )

if not debug:
    # Set everything to None recursively
    defaults = Utils.fill_dict_with_none(defaults)

# Add the hostname to the defaults
defaults['computer'] = socket.gethostname()

# Configure wandb logging
run = wandb.init(
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
    runner = Runner(config=config, tmp_dir=tmp_dir, debug=debug, sweep_id=run.sweep_id)
    runner.run()

    # Close wandb run
    wandb_dir_path = wandb.run.dir
    wandb.join()

    # Delete the local files
    if os.path.exists(wandb_dir_path):
        shutil.rmtree(wandb_dir_path)