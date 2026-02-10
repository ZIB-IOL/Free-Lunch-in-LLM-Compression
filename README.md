## A Free Lunch in LLM Compression: Revisiting Retraining after Pruning

This repository contains the code to reproduce the experiments from the paper ["A Free Lunch in LLM Compression: Revisiting Retraining after Pruning"](https://arxiv.org/abs/2510.14444).
The code is based on [PyTorch 2.8](https://pytorch.org/) and the experiment-tracking platform [Weights & Biases](https://wandb.ai). The results in the paper were generated using [Python 3.12](https://www.python.org/) and the environment defined in [`requirements.txt`](requirements.txt). To evaluate the zero-shot accuracies at the end of a run, additionally install the [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).

### Usage
The [`main.py`](main.py) file starts and configures experiments. The experiments are either configured within this file or via Weights & Biases. When the `--debug` flag is set, the config dictionary inside the file is used, otherwise it is overwritten with a config dictionary provided by Weights & Biases.

The rest of the project is structured as follows:

- [`caching_dummy.py`](caching_dummy.py): Contains a class that downloads and tokenizes the text data sets used for the experiments.
- [`runner.py`](runner.py): Contains a class that prepares the model and data, starts the pruning and reconstruction, and manages fine-tuning.
- [`custom_layers.py`](custom_layers.py): Contains custom modules used for [MaskLoRA](https://github.com/ZIB-IOL/PERP) PEFT after pruning.
- [`peft_methods.py`](peft_methods.py): Contains classes used for applying custom PEFT modules to the pruned model.
- [`prune_methods.py`](prune_methods.py): Contains a pipeline for pruning and reconstructing the model.
- [`prune_flap.py`](prune_flap.py): Contains a pipeline for pruning with [FLAP](https://github.com/CASIA-LMC-Lab/FLAP).
- [`utilities.py`](utilities.py): Contains useful auxiliary functions and classes.

### Citation
In case you find the paper or the implementation useful for your own research, please consider citing:

```
@misc{wagner2026freelunchllmcompression,
      title={A Free Lunch in LLM Compression: Revisiting Retraining after Pruning}, 
      author={Moritz Wagner and Christophe Roux and Max Zimmer and Sebastian Pokutta},
      year={2026},
      eprint={2510.14444},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.14444}, 
}
```