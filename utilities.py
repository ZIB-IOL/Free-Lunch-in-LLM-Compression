import math
import sys
from typing import List, NamedTuple, Optional, Tuple
from torch.optim.optimizer import Optimizer as Optimizer
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn import functional as F
import transformers
import time
import torch.nn.utils.prune as prune
import numpy as np
import os

class PruneUtils:
    """Utilities for pruning"""

    @staticmethod
    def get_n_m_pruning_mask(W_saliency: torch.Tensor, prune_n: int, prune_m: int) -> torch.Tensor:
        """Normal n:m magnitude pruning. Prunes n out of every m weights, using the saliency in W. W should be the metric tensor, i.e. the absolute value tensor in the case of magnitude pruning."""
        W_mask = torch.zeros_like(W_saliency, dtype=torch.bool)
        for ii in range(W_saliency.shape[1]):
            if ii % prune_m == 0:
                tmp = W_saliency[:,ii:(ii+prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        return W_mask

    @staticmethod
    @torch.no_grad()
    def prepare_calibration_input(args, model, dataloader, device, pad_token_id=-100, mask_pad_tokens=False, batch_size=None):
        # if mask_pad_tokens is False and batch size > 1, we collate the batches as in
        # DataCollatorWithFlattening and use position ids for sequence awareness
        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = Utils.get_layerblock_list(model=model, block_size=None)

        if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]

        cache = {'inps': [], 'attention_mask': [], "position_ids": [], 'position_embeddings': [], 'loss_mask': []}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                if hasattr(module, "attention_type"):
                    self.attention_type = module.attention_type
                else:
                    self.attention_type = None
                    
            def forward(self, inp, **kwargs):
                cache['inps'].append(inp.cpu())
                cache['attention_mask'].append(kwargs['attention_mask'].cpu() if kwargs['attention_mask'] is not None else None)
                if 'position_ids' in kwargs:
                    cache['position_ids'].append(kwargs['position_ids'].cpu() if kwargs['position_ids'] is not None else None)
                if 'position_embeddings' in kwargs:
                    cache['position_embeddings'].append((kwargs['position_embeddings'][0].cpu(),
                                                         kwargs['position_embeddings'][1].cpu()) if kwargs['position_embeddings'] is not None else None)
                raise ValueError

        bs = args.batch_size if batch_size is None else batch_size
        layers[0] = Catcher(layers[0])
        for i in range(len(dataloader) // bs):
            batch = [dataloader[i * bs + j] for j in range(bs)]
            if not mask_pad_tokens:
                batch = [b[b != pad_token_id] for b in batch]
                pos_ids = [torch.arange(len(b), device=device) for b in batch]
                pos_ids = torch.cat(pos_ids, dim=0)[None, ...]
                batch = torch.cat(batch, dim=0)[None, ...]
            else:
                pos_ids = None
                batch = torch.cat(batch, dim=0)
            try:
                model(batch.to(device), attention_mask=batch != pad_token_id if mask_pad_tokens else None, position_ids=pos_ids.to(device) if pos_ids is not None else None)
            except ValueError:
                pass
            cache['loss_mask'].append(batch.cpu() != pad_token_id if mask_pad_tokens else None)
        layers[0] = layers[0].module

        inps = cache['inps']
        loss_mask = cache['loss_mask']
        outs = [inp.clone() for inp in inps]
        attention_mask = cache['attention_mask'] if len(cache['attention_mask']) > 0 else None
        position_ids = cache['position_ids'] if len(cache['position_ids']) > 0 else None
        position_embeddings = cache['position_embeddings'] if len(cache['position_embeddings']) > 0 else None
        model.config.use_cache = use_cache
        return inps, outs, attention_mask, position_ids, position_embeddings, loss_mask


    @staticmethod
    @torch.no_grad()
    def eval_reconstruction_error(args: NamedTuple, layer: torch.nn.Module, inps: torch.Tensor, outs: torch.Tensor, device: torch.device,
                                  attention_mask: Optional[torch.Tensor], position_ids: Optional[torch.Tensor],
                                  position_embeddings: Optional[torch.Tensor], loss_mask: Optional[torch.Tensor]) -> float:
        tensor_dataloader = Utils.get_list_dataloader(args, (inps, outs))
        other_loader = Utils.get_list_dataloader(args,
                                                 (attention_mask if attention_mask is not None else [None] * len(inps),
                                                  position_ids if position_ids is not None else [None] * len(inps),
                                                  position_embeddings if position_embeddings is not None else [None] * len(inps)))
        if loss_mask is not None:
            loss_mask_loader = Utils.get_tensor_dataloader(args, (loss_mask,))
        else:
            loss_mask_loader = [None for _ in range(len(tensor_dataloader))]
        criterion = nn.MSELoss(reduction="mean").cuda()
        layer.eval()

        ret_loss = 0.
        for (inputs, outps), (amask, pids, pembeds), loss_mask in zip(tensor_dataloader, other_loader, loss_mask_loader):
            kwargs = {}
            if amask is not None:
                kwargs['attention_mask'] = amask.to(device)
            if pids is not None:
                kwargs['position_ids'] = pids.to(device)
            if pembeds is not None:
                kwargs['position_embeddings'] = (pembeds[0].to(device), pembeds[1].to(device))
            with torch.amp.autocast('cuda'):
                outputs = layer(inputs.to(device), **kwargs)[0]
                if loss_mask is not None:
                    loss = criterion(outputs[loss_mask], outps[loss_mask].to(device))
                else:
                    loss = criterion(outputs, outps.to(device))

            ret_loss += loss.item() * len(inputs)
        return ret_loss / len(inps)

class SequentialLayerBlock(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = nn.ModuleList(layer_list)
        
    
    def forward(self, x, **kwargs):
        for layer_idx, layer in enumerate(self.layer_list):
            x = layer(x, **kwargs)
            if (layer_idx < len(self.layer_list) - 1) and isinstance(x, tuple):
                # In case of the last layer, we want to return the tuple
                x = x[0]
        return x

class ListDataset(Dataset[tuple[List[Tensor | None], ...]]):
    """Dataset wrapping lists of tensors or None. We use this mainly for making dataloaders of attention masks,
    position ids, and position embeddings, as depending on the config and model, these might be None.
    The collate function handles the case where a list can contain both tensors and None as can be the case for
    attention masks when using flash attention 2.
    This is a bit of a hack, but it works for our use case."""
    lists: tuple[List[Tensor | None], ...]
    def __init__(self, *lists: List[Tensor | None]) -> None:
        assert all(len(lists[0]) == len(l) for l in lists), "Size mismatch between lists"
        self.lists = lists

    def __getitem__(self, index):
        return tuple(l[index] for l in self.lists)

    def __len__(self):
        return len(self.lists[0])
    
    def get_collate_fn(self):
        """Return a collate function that takes a sequence of tuples of tensors to collate into a batch.
        For every tuple index, it replaces None with an all-ones tensor of the same shape as the other
        tensors and returns None if all tensors are None. That is, it returns a tuple containing batch
        tensors or None at every index."""
        def collate_fn(batch):
            nones = [0 for _ in range(len(batch[0]))]
            shapes, dtypes, devices = [None for _ in range(len(batch[0]))], [None for _ in range(len(batch[0]))], [None for _ in range(len(batch[0]))]
            for i in range(len(batch)):
                for j in range(len(batch[i])):
                    if batch[i][j] is None:
                        nones[j] += 1
                    else:
                        # we only need this information in case we have some None attention masks in the batch.
                        # other cases where batch[i][j] is not a tensor are not relevant in our use case.
                        shapes[j] = batch[i][j].shape if torch.is_tensor(batch[i][j]) else None
                        dtypes[j] = batch[i][j].dtype if torch.is_tensor(batch[i][j]) else None
                        devices[j] = batch[i][j].device if torch.is_tensor(batch[i][j]) else None
            if all(nones_ == len(batch) for nones_ in nones):
                return tuple([None] * len(batch[0])) if len(batch) > 1 else None
            
            for j in range(len(batch[0])):
                if nones[j] != 0 and nones[j] != len(batch):
                    for i in range(len(batch)): # if the attention is not None for all elements, we replace the None with a tensor of ones
                        batch[i] = tuple([batch[i][j_] if batch[i][j_] is not None or j_ != j else torch.ones(shapes[j], dtype=dtypes[j], device=devices[j]) for j_ in range(len(batch[i]))])

            collated = []
            for j in range(len(batch[0])):
                if nones[j] != len(batch):
                    collated.append(torch.utils.data.default_collate([batch[i][j] for i in range(len(batch))]))
                else:
                    collated.append(None)

            if len(collated) == 1:
                return collated[0]
            return collated
        return collate_fn


class Utils:
    @staticmethod
    @torch.no_grad()
    def get_outputs(args: NamedTuple, layer: torch.nn.Module, inps: torch.Tensor, outs: torch.Tensor,
                    attention_mask: Optional[torch.Tensor], position_ids: Optional[torch.Tensor], position_embeddings: Optional[torch.Tensor],
                    loss_mask: Optional[torch.Tensor], device: torch.device):
        zip_list = [
            inps,
            attention_mask if attention_mask is not None else [None] * len(inps),
            position_ids if position_ids is not None else [None] * len(inps),
            position_embeddings if position_embeddings is not None else [None] * len(inps),
            loss_mask if loss_mask is not None else [None] * len(inps),
        ]
        for j, (inp, amask, pids, pembeds, lmask) in enumerate(zip(*zip_list)):
            kwargs = {}
            if amask is not None:
                kwargs['attention_mask'] = amask.to(device)
            if pids is not None:
                kwargs['position_ids'] = pids.to(device)
            if pembeds is not None:
                kwargs['position_embeddings'] = (pembeds[0].to(device), pembeds[1].to(device))
            out = layer(inp.to(device), **kwargs)
            if isinstance(out, tuple):
                out = out[0]            
            if (new_shape := outs[j].shape) != out.shape and outs[j].numel() == out.numel():
                out = out.reshape(new_shape)
            outs[j] = out.cpu()

        return outs

    @staticmethod
    def get_tensor_dataloader(args, tensors, batch_size=None) -> torch.utils.data.DataLoader:
        """Return a TensorDataLoader for the given input and output tensors."""
        tensor_dataset = torch.utils.data.TensorDataset(*tensors)
        tensor_dataloader = torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=batch_size if batch_size is not None else args.batch_size, 
            shuffle=False,
            num_workers=0,      # Probably makes no difference as we are using already initialized tensors
            pin_memory=False,   # Should not be any need since we are using already initialized tensors
        )
        return tensor_dataloader
    
    @staticmethod
    def get_list_dataloader(args, lists, batch_size=None) -> torch.utils.data.DataLoader:
        """Return a TensorDataLoader for the given input and output tensors."""
        list_dataset = ListDataset(*lists)
        list_dataloader = torch.utils.data.DataLoader(
            list_dataset,
            batch_size=batch_size if batch_size is not None else args.batch_size, 
            shuffle=False,
            num_workers=0,      # Probably makes no difference as we are using already initialized tensors
            pin_memory=False,   # Should not be any need since we are using already initialized tensors
            collate_fn=list_dataset.get_collate_fn()
        )
        return list_dataloader

    @staticmethod
    def change_model_state(model, train: bool):
        """Change the model state to train or eval mode."""
        sys.stdout.write(f"Changing model state to {'train' if train else 'eval'} mode.\n")
        if train:
            model.train()
            model.enable_input_require_grads() # Needed for PEFT: https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
        else:
            model.eval()

    @staticmethod
    def compute_ppl_wikitext(model, testenc, bs=1, device=None):
        # Get input IDs
        testenc = testenc.input_ids

        # Calculate number of samples
        nsamples = testenc.numel() // model.seqlen

        # List to store negative log likelihoods
        nlls = []
        #sys.stdout.write(f"Number of samples: {nsamples}")

        # Loop through each batch
        for i in tqdm(range(0, nsamples, bs)):
            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to device
            inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
            inputs = inputs.reshape(j - i, model.seqlen)

            # Forward pass through the model
            lm_logits = model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * model.seqlen * (j - i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

        sys.stdout.write(f"Computing perplexity for {nsamples} samples.\n")
        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

        sys.stdout.write(f"Perplexity: {ppl.item()} - Emptying CUDA Cache.\n")
        # Empty CUDA cache to save memory
        torch.cuda.empty_cache()
        sys.stdout.write(f"Emptying CUDA Cache done.\n")

        return ppl.item()

    @staticmethod
    @torch.no_grad()
    def evaluate_loss(inps, outs, attention_mask, position_ids, position_embeddings, loss_mask, device, layer, criterion, args):
        zip_list = [
            inps,
            outs,
            attention_mask if attention_mask is not None else [None] * len(inps),
            position_ids if position_ids is not None else [None] * len(inps),
            position_embeddings if position_embeddings is not None else [None] * len(inps),
            loss_mask
        ]
        accumloss = 0
        n_samples = 0
        for inp, out, amask, pids, pembeds, lmask in zip(*zip_list):
            kwargs = {}
            if amask is not None:
                kwargs['attention_mask'] = amask.to(device)
            if pids is not None:
                kwargs['position_ids'] = pids.to(device)
            if pembeds is not None:
                kwargs['position_embeddings'] = (pembeds[0].to(device), pembeds[1].to(device))
            with torch.amp.autocast('cuda'):
                outputs = layer(inp.to(device), **kwargs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.shape != out.shape:
                outputs = outputs.reshape(out.shape)
            out = out.to(outputs.device)
            if lmask is not None and args.mask_pad_tokens:
                lmask = lmask.to(outputs.device)
                if outputs.shape[:len(lmask.shape)] != lmask.shape:
                    if lmask.numel() == outputs.shape[0]:
                        lmask = lmask.flatten()
                    else:
                        outputs = outputs.reshape((*lmask.shape, -1))
                        out = out.reshape((*lmask.shape, -1))
                loss = criterion(outputs[lmask], out[lmask])
            else:
                loss = criterion(outputs, out)
        
            accumloss += loss.item()
            n_samples += 1
        return accumloss / n_samples
    
    
    @staticmethod
    def get_c4_for_calibration(nsamples: int, seed: int, seqlen: int, tokenized_dataset, tokenizer, split: str = "train", allow_smaller_seqlen: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get calibration samples from pre-tokenized C4 dataset.
        
        Args:
            nsamples: Number of samples to generate
            seed: Random seed
            seqlen: Sequence length for each sample
            tokenized_dataset: Pre-tokenized C4 dataset -> we just use this to infer the dataset, we tokenize again to not have the truncation/padding issues
            tokenizer: HuggingFace tokenizer with defined pad_token_id
        """
        traindata = tokenized_dataset[split]
        seeded_generator = torch.Generator(device=torch.device('cpu'))
        seeded_generator.manual_seed(seed)
        trainloader = []

        goal_seqlen = seqlen

        for _ in range(nsamples):
            max_iter = 200000
            it = 0
            while True:
                seqlen = goal_seqlen
                # Sample a random sequence
                i = torch.randint(0, len(traindata), (1,), generator=seeded_generator)
                # Tokenize the sequence
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                # We have found a sequence that is long enough and must not be padded
                if trainenc.input_ids.shape[1] > seqlen:
                    break
                elif allow_smaller_seqlen:
                    seqlen = trainenc.input_ids.shape[1] - 1
                    break
                it += 1
                if it > max_iter:
                    raise ValueError("Could not find long enough sequence. If the model is Mistral, we might need to reduce the sequence length (its >30k).")
            # Sample a random start point for the subsequence
            i = torch.randint(0, max(1, trainenc.input_ids.shape[1] - seqlen - 1), (1,), generator=seeded_generator)
            j = i + seqlen  # This is the end index
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)

        return trainloader
    

    @staticmethod
    def get_c4_for_calibration_no_filter(nsamples: int, seed: int, tokenized_dataset, tokenizer, runner, split: str = "train") -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get calibration samples from pre-tokenized C4 dataset.
        
        Args:
            nsamples: Number of samples to generate
            seed: Random seed
            tokenized_dataset: Pre-tokenized C4 dataset -> we just use this to infer the dataset, we tokenize again to not have the truncation/padding issues
            tokenizer: HuggingFace tokenizer with defined pad_token_id
        """
        if split == "train":
            tokenized_dataset['train'] = tokenized_dataset['train'].shuffle(seed=seed).select(range(nsamples))
        trainer = runner.get_trainer({'train': tokenized_dataset[split], 'validation': None}, batch_size=1, seed=seed)
        dataloader = trainer.get_train_dataloader()
        trainloader = []
        for i, batch in enumerate(dataloader):
            if i >= nsamples:
                break
            inp = batch['input_ids']
            trainloader.append(inp)
        return trainloader


    @staticmethod
    def get_percentage_of_trainable_parameters(model):
        n_params_total = sum(p.numel() for p in model.parameters())
        n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return 100 * float(n_params_trainable) / n_params_total


    @staticmethod
    def fill_dict_with_none(d):
        for key in d:
            if isinstance(d[key], dict):
                Utils.fill_dict_with_none(d[key])  # Recursive call for nested dictionaries
            else:
                d[key] = None
        return d
    

    @staticmethod
    def update_config_with_default(configDict, defaultDict):
        """Update config with default values recursively."""
        for key, default_value in defaultDict.items():
            if key not in configDict:
                configDict[key] = default_value
            elif isinstance(default_value, dict):
                configDict[key] = Utils.update_config_with_default(configDict.get(key, {}), default_value)
        return configDict

    
    @staticmethod
    def eval_zero_shot(model_name, model, task_list):
        from lm_eval import evaluator, models
        import lm_eval.models.huggingface

        # Get the task dict directly from the provided task list
        #available_tasks = get_task_dict(task_list)

        model_args = f"pretrained={model_name},cache_dir=./llm_weights"
        limit = None 
        if "70b" in model_name or "65b" in model_name:
            limit = 2000

        model = models.huggingface.HFLM(pretrained=model)

        results = evaluator.simple_evaluate(
            model=model,
            model_args=model_args,  # Now passing as dict instead of string
            tasks=task_list,
            num_fewshot=0,
            batch_size=None,
            device=None,
            limit=limit,
            check_integrity=False,
        )

        # Clean up the results, since some keys might end with ,0 or ,none
        cleaned_results = {'results': {}}
        for task in results['results'].keys():
            cleaned_results['results'][task] = {}
            for key, value in results['results'][task].items():
                clean_key = key.replace(",0", "").replace(",none", "")
                cleaned_results['results'][task][clean_key] = value
        results = cleaned_results

        return results

    @staticmethod
    def _clean_results(results):
        """
        Remove lm-eval suffixes like ',0' or ',none' from metric keys.
        """
        cleaned_results = {"results": {}}
        for task, task_metrics in results["results"].items():
            cleaned_results["results"][task] = {}
            for key, value in task_metrics.items():
                clean_key = key.replace(",0", "").replace(",none", "")
                cleaned_results["results"][task][clean_key] = value
        return cleaned_results

    @staticmethod
    def _extract_metric(task_metrics, metric_name, preferred_filters=None):
        """
        Robustly extract a metric from lm-eval output.

        Examples of keys you may see:
          - 'pass@1'
          - 'pass@1_stderr'
          - 'pass@1,create_test'
          - 'exact_match,strict-match'
          - 'exact_match_stderr,flexible-extract'
        """
        preferred_filters = preferred_filters or []

        # 1) exact key
        if metric_name in task_metrics:
            return task_metrics[metric_name]

        # 2) key with preferred filters
        for filt in preferred_filters:
            k = f"{metric_name},{filt}"
            if k in task_metrics:
                return task_metrics[k]

        # 3) first key that starts with metric_name + comma
        candidates = [k for k in task_metrics if k == metric_name or k.startswith(metric_name + ",")]
        if candidates:
            return task_metrics[candidates[0]]

        raise KeyError(f"Could not find metric '{metric_name}' in keys: {list(task_metrics.keys())}")

    @staticmethod
    def _extract_stderr(task_metrics, metric_name, preferred_filters=None):
        """
        Robustly extract stderr for a metric from lm-eval output.
        """
        preferred_filters = preferred_filters or []
        stderr_name = f"{metric_name}_stderr"

        # 1) exact key
        if stderr_name in task_metrics:
            return task_metrics[stderr_name]

        # 2) key with preferred filters
        for filt in preferred_filters:
            k = f"{stderr_name},{filt}"
            if k in task_metrics:
                return task_metrics[k]

        # 3) first key that starts with stderr_name + comma
        candidates = [k for k in task_metrics if k == stderr_name or k.startswith(stderr_name + ",")]
        if candidates:
            return task_metrics[candidates[0]]

        return np.nan  # some tasks may not report stderr

    @staticmethod
    def eval_reasoning(
        model_name,
        model,
        task_list,
        gsm8k_variant="gsm8k",
        limit=None,
    ):
        """
        Evaluate reasoning/code tasks with lm-evaluation-harness.

        Args:
            model_name: string model identifier
            model: loaded HF model object
            task_list: e.g. ["humaneval", "gsm8k"]
            gsm8k_variant: "gsm8k" or "gsm8k_cot"
            gsm8k_num_fewshot: override few-shot for GSM8K. Use 0 for zero-shot.
            limit: optional example limit
        """
        from lm_eval import evaluator, models
        import lm_eval.models.huggingface

        # HumanEval executes generated code.
        # HF's code-eval metric requires this env var before running.
        if any("humaneval" in t for t in task_list):
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        model_args = f"pretrained={model_name},cache_dir=./llm_weights"

        lm = models.huggingface.HFLM(pretrained=model)

        # Per-task overrides
        task_manager = None

        # If you want gsm8k_cot instead of gsm8k, swap it here.
        resolved_tasks = []
        for t in task_list:
            if t == "gsm8k":
                resolved_tasks.append(gsm8k_variant)
            else:
                resolved_tasks.append(t)

        # lm-eval accepts a single num_fewshot for the run.
        # Since HumanEval is 0-shot anyway, using 0 here is fine if your list is
        # ["humaneval", "gsm8k"] and you want zero-shot GSM8K.

        results = evaluator.simple_evaluate(
            model=lm,
            model_args=model_args,
            tasks=resolved_tasks,
            batch_size=None,
            device=None,
            check_integrity=False,
            log_samples=True,                 # useful for generative tasks
            confirm_run_unsafe_code=True,    # required for HumanEval
        )

        return Utils._clean_results(results)


    @staticmethod
    def get_layers_of_modules(module, layers=[nn.Linear], name='') -> dict:
        """
        Recursively find the layers of a certain type in a module.

        Args:
            module (nn.Module): PyTorch module.
            layers (list): List of layer types to find.
            name (str): Name of the module.

        Returns:
            dict: Dictionary of layers of the given type(s) within the module.
        """
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            if "lm_head" in name1: # we do not want to prune and reconstruct the lm_head
                continue
            res.update(Utils.get_layers_of_modules(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res
    

    @staticmethod
    def get_layerblock_list(model: torch.nn.Module, block_size: int | None = 1, full_model: bool = False):
        base_model = model
        while hasattr(base_model, "model"):
            base_model = base_model.model
        
        if hasattr(base_model, 'decoder'):
            # OPT CASE
            layer_list = base_model.decoder.layers
        elif hasattr(base_model, 'layers'):
            # LLAMA CASE
            layer_list = base_model.layers
        else:
            raise ValueError("Model does not have a 'decoder' or 'layers' attribute.")
        
        if full_model:
            return [Utils.join_sequential_layers(layer_list)]
        
        if block_size is None:
            block_size = 1
        assert float(block_size).is_integer(), "Block size must be an integer."
        assert 0 < block_size <= len(layer_list), "Block size must be between 1 and the number of layers."

        if block_size > 1:
            layer_list = [Utils.join_sequential_layers(layer_list[i:i+block_size]) for i in range(0, len(layer_list), block_size)]        
        return layer_list
    

    @staticmethod
    def get_reconstruction_layers(model, block_size=1, rec=False, runner=None):
        assert isinstance(block_size, int), "block_size must be an integer"
        # one or more transformer blocks per submodel
        if block_size > 0:
            return Utils.get_layerblock_list(model=model, block_size=block_size)
        # attention and MLP as separate submodels
        elif block_size == -1:
            if runner is None or runner.is_opt:
                aw_class = AttnWrapper
                mw_class = MLPWrapper
            elif runner.is_llama:
                aw_class = AttnWrapperLlama
                mw_class = MLPWrapperLlama
            elif runner.is_qwen:
                aw_class = AttnWrapperQwen
                mw_class = MLPWrapperQwen
            else:
                raise ValueError("Invalid architecture for block size -1.")
            layer_list = Utils.get_layerblock_list(model=model, block_size=1)
            wrapper_list = []
            for layer in layer_list:
                wrapper_list.append(aw_class(layer))
                wrapper_list.append(mw_class(layer))
            return wrapper_list
        # each matrix as its own submodel
        elif block_size == -2:
            layer_list = Utils.get_layerblock_list(model=model, block_size=1)
            wrapper_list = []
            for layer in layer_list:
                if runner is None or runner.is_opt:
                    wrapper_class = AttnWrapperPerMatrix
                    wrapper_list.append(wrapper_class(layer, matrix_names="q_proj", last_in_block=False))
                    wrapper_list.append(wrapper_class(layer, matrix_names="k_proj", last_in_block=False))
                    wrapper_list.append(wrapper_class(layer, matrix_names="v_proj", last_in_block=False))
                    wrapper_list.append(wrapper_class(layer, matrix_names="out_proj", last_in_block=True))
                    wrapper_class = MLPWrapperPerMatrix
                    wrapper_list.append(wrapper_class(layer, matrix_name="fc1", last_in_block=False))
                    wrapper_list.append(wrapper_class(layer, matrix_name="fc2", last_in_block=True))
                else:
                    wrapper_class = AttnWrapperPerMatrixLlama
                    wrapper_list.append(wrapper_class(layer, matrix_names="q_proj", last_in_block=False))
                    wrapper_list.append(wrapper_class(layer, matrix_names="k_proj", last_in_block=False))
                    wrapper_list.append(wrapper_class(layer, matrix_names="v_proj", last_in_block=False))
                    wrapper_list.append(wrapper_class(layer, matrix_names="o_proj", last_in_block=True))
                    wrapper_class = MLPWrapperPerMatrixLlama
                    wrapper_list.append(wrapper_class(layer, matrix_name="gate_proj", last_in_block=False))
                    wrapper_list.append(wrapper_class(layer, matrix_name="up_proj", last_in_block=False))
                    wrapper_list.append(wrapper_class(layer, matrix_name="down_proj", last_in_block=True))
            return wrapper_list
        elif block_size == -4:
            layer_list = Utils.get_layerblock_list(model=model, block_size=1)
            wrapper_list = []
            for layer in layer_list:
                wrapper_class = AttnWrapperPerMatrixLlama
                wrapper_list.append(wrapper_class(layer, matrix_names="k_proj", last_in_block=True))
                wrapper_class = MLPWrapperPerMatrixLlama
                wrapper_list.append(wrapper_class(layer, matrix_name="up_proj", last_in_block=True))
            return wrapper_list
        elif block_size == -5:
            layer_list = Utils.get_layerblock_list(model=model, block_size=1)
            wrapper_list = []
            for layer in layer_list:
                wrapper_list.append(KAndUpWrapper(layer))
            return wrapper_list
        else:
            raise ValueError("Invalid block_size")

    
    @staticmethod
    def join_sequential_layers(layer_list: list):
        """Returns a module that simply sequentially forwards through the layers in layer_list."""
        # Note: We cannot use torch.nn.Sequential here, since its forward does not expect additional kwargs
        
        return SequentialLayerBlock(layer_list)
   

    @staticmethod
    def combine_pruned_layers(pruned_layers):
        """Combine a list of pruned layers into a single module."""
        if any([isinstance(layer, AttnWrapper) or isinstance(layer, MLPWrapper) for layer in pruned_layers]):
            raise NotImplementedError("Different pruning and reconstruction block sizes are not yet supported for sub-transformer-block pruning.")
        layer_list = []
        for layer in pruned_layers:
            if hasattr(layer, 'layer_list'):
                layer_list.extend(layer.layer_list)
            else:
                layer_list.append(layer)
        return Utils.join_sequential_layers(layer_list)
    

    @staticmethod
    def check_sparsity(model):
        use_cache = model.config.use_cache 
        model.config.use_cache = False 

        layers = Utils.get_layerblock_list(model=model)
        
        count = 0 
        total_params = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = Utils.get_layers_of_modules(layer)

            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W==0).sum().item()
                total_params += W.numel()

                sub_count += (W==0).sum().item()
                sub_params += W.numel()

            sys.stdout.write(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}.\n")

        model.config.use_cache = use_cache 
        return float(count)/total_params
    

    @staticmethod
    def get_encoder(model):
        base_model = model
        while hasattr(base_model, "model"):
            base_model = base_model.model
        first_block = base_model.decoder.layers[0]

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.kwargs = {}
                self.inp = None

            def forward(self, inp, **kwargs):
                if isinstance(inp, tuple):
                    self.inp = inp[0]
                else:
                    self.inp = inp
                self.kwargs = {}
                self.kwargs['attention_mask'] = kwargs['attention_mask']
                if 'position_ids' in kwargs:
                    self.kwargs['position_ids'] = kwargs['position_ids']
                if 'position_embeddings' in kwargs:
                    self.kwargs['position_embeddings'] = kwargs['position_embeddings']
                raise ValueError
            
        class Encoder(nn.Module):
            def __init__(self, model: torch.nn.Module, first_block: Catcher):
                super().__init__()
                self.model = model
                self.first_block = first_block

            def forward(self, inp, **kwargs):
                use_cache = self.model.config.use_cache
                self.model.config.use_cache = False
                self.model.decoder.layers[0] = self.first_block
                try:
                    _ = self.model(inp, **kwargs)
                except ValueError:
                    pass
                self.model.decoder.layers[0] = self.first_block.module
                self.model.config.use_cache = use_cache
                return self.first_block.inp, self.first_block.kwargs

        return Encoder(base_model, Catcher(first_block))


    @staticmethod
    def recursive_to_device(obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: Utils.recursive_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Utils.recursive_to_device(v, device) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(Utils.recursive_to_device(v, device) for v in obj)
        else:
            return obj



class SelectiveMethods:
    """Class of methods that can be used to selectively activate and deactivate certain layers."""

    @staticmethod
    def activate_model(model: torch.nn.Module):
        """Activate the entire model"""
        for param in model.parameters():
            param.requires_grad_(True)

    @staticmethod
    def deactivate_model(model: torch.nn.Module):
        """Deactivate the entire model"""
        for param in model.parameters():
            param.requires_grad_(False)


    @staticmethod
    def activate_specific_modules(moduleList: List[torch.nn.Module]):
        """Activate specific modules"""
        for module in moduleList:
            for param in module.parameters():
                param.requires_grad_(True)

    @staticmethod
    def change_biases_activation(model: torch.nn.Module, activate: bool):
        """Change the bias of a model to be trainable or not"""
        for module in model.modules():
            if hasattr(module, 'bias') and not isinstance(module.bias, type(None)):
                module.bias.requires_grad_(activate)

    @staticmethod
    def deactivate_biases(model: torch.nn.Module):
        """Deactivate the bias of a model"""
        SelectiveMethods.change_biases_activation(model, False)

    @staticmethod
    def activate_biases(model: torch.nn.Module):
        """Activate the bias of a model"""
        SelectiveMethods.change_biases_activation(model, True)

    @staticmethod
    def change_layer_norm_activation(model: torch.nn.Module, activate: bool):
        """Change the layer params of a model to be trainable or not"""
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm) or isinstance(module, transformers.models.mistral.modeling_mistral.MistralRMSNorm) or isinstance(module, transformers.models.llama.modeling_llama.LlamaRMSNorm) or isinstance(module, transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm):
                # Includes RMSNorms from Mistral and LLAMA
                module.requires_grad_(requires_grad=activate)

    @staticmethod
    def deactivate_layer_norm_params(model: torch.nn.Module):
        """Deactivate the layer norm params of a model"""
        SelectiveMethods.change_layer_norm_activation(model, False)

    @staticmethod
    def activate_layer_norm_params(model: torch.nn.Module):
        """Activate the layer norm params of a model"""
        SelectiveMethods.change_layer_norm_activation(model, True)


class WandaWrapper:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples


class SparseGPTWrapper:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


class FLAPWrapper:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, metric):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.out_dim = layer.weight.data.shape[0]
        self.in_dim = layer.weight.data.shape[1]
        self.type = metric
        self.nsamples = 0

        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        if self.type == "WIFN":
            self.scaler_inp = torch.zeros((self.in_dim), device=self.dev)
        else:   
            self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (dim, seqlen)

        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.type == "WIFN":
            inp = inp.type(torch.float32)
            self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
            self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / (self.nsamples + batch_size)
        else:
            if self.nsamples == 0:
                self.fluc_inp = 0
            else:
                self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号

        self.nsamples += batch_size

        
    def free(self):
        self.baseline_inp = None
        if hasattr(self, 'fluc_inp'):
            self.fluc_inp = None
        if hasattr(self, 'scaler_inp'):
            self.scaler_inp = None
        torch.cuda.empty_cache()  


# wrap transformer blocks to only expose certain parts

class KAndUpWrapper(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        # hide the module from methods like .parameters() by wrapping it in a tuple
        self.module = (module,)
        self.k_proj = module.self_attn.k_proj
        self.up_proj = module.mlp.up_proj
        self.hook_handle = None
        self.pass_through = False
        self.last_in_block = True
        self.output_activation = None
    
    def get_hook(self):
        def hook_fn(module, inp, out):
            assert self.output_activation is None, "Output activation already set."
            self.output_activation = out
            raise ValueError
        return hook_fn

    def register_hook(self):
        assert self.hook_handle is None, "Hook already registered."
        self.hook_handle = self.up_proj.register_forward_hook(self.get_hook())

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def activate_pass_through(self):
        self.pass_through = True
        self.remove_hook()

    def deactivate_pass_through(self):
        self.pass_through = False

    def forward(self, x: torch.Tensor, **kwargs):
        if not self.pass_through:
            self.register_hook()
        try:
            out = self.module[0](x, **kwargs)
        except ValueError:
            out = self.output_activation
        self.remove_hook()
        self.output_activation = None
        return out


class AttnWrapper(nn.Module):
    def __init__(self, module: torch.nn.Module, set_submodules: bool = True):
        super().__init__()
        # hide the module from methods like .parameters() by wrapping it in a tuple
        self.module = (module,)
        if set_submodules:
            # we set the submodules of the subblock we want to use as attributes so
            # they are found by methods like .parameters()
            self.set_submodules(module)
    
    def set_submodules(self, module: torch.nn.Module):
        self.self_attn = module.self_attn
        self.self_attn_layer_norm = module.self_attn_layer_norm
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        past_key_value: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        position_ids: torch.LongTensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.module[0].do_layer_norm_before:
            hidden_states = self.module[0].self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.module[0].self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            position_ids=position_ids,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.module[0].dropout, training=self.module[0].training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.module[0].do_layer_norm_before:
            hidden_states = self.module[0].self_attn_layer_norm(hidden_states)

        return (hidden_states,)
    

class AttnWrapperLlama(nn.Module):
    def __init__(self, module: torch.nn.Module, set_submodules: bool = True):
        super().__init__()
        self.module = (module,)
        if set_submodules:
            self.set_submodules(module)
    
    def set_submodules(self, module: torch.nn.Module):
        self.self_attn = module.self_attn
        self.input_layernorm = module.input_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[transformers.cache_utils.Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: transformers.processing_utils.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.module[0].input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.module[0].self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if residual.device != hidden_states.device:
            hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states
        return hidden_states


class AttnWrapperQwen(nn.Module):
    def __init__(self, module: torch.nn.Module, set_submodules: bool = True):
        super().__init__()
        self.module = (module,)
        if set_submodules:
            self.set_submodules(module)

    def set_submodules(self, module: torch.nn.Module):
        self.self_attn = module.self_attn
        self.input_layernorm = module.input_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: transformers.processing_utils.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.module[0].self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if residual.device != hidden_states.device:
            hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states
        return hidden_states


class AttnWrapperPerMatrix(AttnWrapper):
    def __init__(self, module: torch.nn.Module, matrix_names: str | list, propagate_to: str | None = None, last_in_block: bool = False,
                 include_ln: bool = False):
        if isinstance(matrix_names, str):
            matrix_names = [matrix_names]
        assert all(matrix_name in ["q_proj", "k_proj", "v_proj", "out_proj"] for matrix_name in matrix_names), "Invalid matrix name."
        self.matrix_names = matrix_names
        self.propagate_to = propagate_to
        self.output_activation = None
        self.output_hook_handles = []
        self.last_in_block = last_in_block # if true, we propagate the calibration data after this matrix has been pruned
                                           # e.g. after pruning the out_proj, we propagate the calibration so we can
                                           # continue with reconstructing the MLP
        super().__init__(module, set_submodules=False)
        self.set_submodules(module, (matrix_names + ["self_attn_layer_norm"]) if include_ln else matrix_names)
        self.pass_through = False # if true, we pass the data through the whole module so we can compute the
                                  # calibration data for the next module
        self.batch_size = None # save batch size of the input tensor to reshape the output caught by the hook
        
    def set_submodules(self, module: torch.nn.Module, matrix_names: list):
        for matrix_name in matrix_names:
            if hasattr(module, matrix_name):
                setattr(self, matrix_name, getattr(module, matrix_name))
            else:
                setattr(self, matrix_name, getattr(module.self_attn, matrix_name))

    def register_hook(self):
        if not self.pass_through:
            if self.propagate_to is None:
                for matrix_name in self.matrix_names:
                    self.output_hook_handles.append(getattr(self, matrix_name).register_forward_hook(self.output_hook_post))
            else:
                if hasattr(self.module[0], self.propagate_to):
                    self.output_hook_handles.append(getattr(self.module[0], self.propagate_to).register_forward_pre_hook(self.output_hook_pre))
                else:
                    self.output_hook_handles.append(getattr(self.module[0].self_attn, self.propagate_to).register_forward_pre_hook(self.output_hook_pre))
    
    def remove_hook(self):
        for handle in self.output_hook_handles:
            handle.remove()
        self.output_hook_handles = []
    
    def output_hook_post(self, module, inp, out):
        # save the output of the submodule to return it as the output of the forward method
        if isinstance(out, tuple):
            out = out[0]
        if out.size(0) != self.batch_size:
            out = out.reshape([self.batch_size, -1] + list(out.shape[1:]))
        if self.output_activation is None:
            self.output_activation = [out]
        else:
            self.output_activation.append(out)
        if len(self.output_activation) == len(self.matrix_names):
            raise ValueError() # we raise an error to stop the forward pass and return the output caught by the hook

    def output_hook_pre(self, module, inp):
        # save the input of the submodule to return it as the output of the forward method
        if isinstance(inp, tuple):
            inp = inp[0]
        if inp.size(0) != self.batch_size:
            inp = inp.reshape([self.batch_size, -1] + list(inp.shape[1:]))
        self.output_activation = (inp,)
        raise ValueError() # we raise an error to stop the forward pass and return the output caught by the hook

    def activate_pass_through(self):
        self.pass_through = True
        self.remove_hook()

    def deactivate_pass_through(self):
        self.pass_through = False

    def forward(self, *args, **kwargs):
        self.batch_size = args[0].shape[0]
        self.register_hook()
        if self.output_hook_handles is not None and len(self.output_hook_handles) > 0:
            try:
                out = super().forward(*args, **kwargs)
            except ValueError:
                pass
            if len(self.output_activation) == 1:
                out = self.output_activation[0]
            else:
                out = torch.cat(self.output_activation, dim=1)
        else:
            out = super().forward(*args, **kwargs)
        self.remove_hook()
        self.output_activation = None
        self.batch_size = None
        return out


class AttnWrapperPerMatrixLlama(AttnWrapperLlama):
    def __init__(self, module: torch.nn.Module, matrix_names: str | list, last_in_block: bool = False):
        if isinstance(matrix_names, str):
            matrix_names = [matrix_names]
        assert all(matrix_name in ["q_proj", "k_proj", "v_proj", "o_proj"] for matrix_name in matrix_names), "Invalid matrix name."
        self.matrix_names = matrix_names
        self.output_activation = None
        self.output_hook_handles = []
        self.last_in_block = last_in_block # if true, we propagate the calibration data after this matrix has been pruned
                                           # e.g. after pruning the out_proj, we propagate the calibration so we can
                                           # continue with reconstructing the MLP
        super().__init__(module, set_submodules=False)
        self.set_submodules(module, matrix_names)
        self.pass_through = False # if true, we pass the data through the whole module so we can compute the
                                  # calibration data for the next module
        self.batch_size = None # save batch size of the input tensor to reshape the output caught by the hook
        
    def set_submodules(self, module: torch.nn.Module, matrix_names: list):
        for matrix_name in matrix_names:
            if hasattr(module, matrix_name):
                setattr(self, matrix_name, getattr(module, matrix_name))
            else:
                setattr(self, matrix_name, getattr(module.self_attn, matrix_name))

    def register_hook(self):
        if not self.pass_through:
            for matrix_name in self.matrix_names:
                self.output_hook_handles.append(getattr(self, matrix_name).register_forward_hook(self.output_hook_post))
    
    def remove_hook(self):
        for handle in self.output_hook_handles:
            handle.remove()
        self.output_hook_handles = []
    
    def output_hook_post(self, module, inp, out):
        # save the output of the submodule to return it as the output of the forward method
        if isinstance(out, tuple):
            out = out[0]
        if out.size(0) != self.batch_size:
            out = out.reshape([self.batch_size, -1] + list(out.shape[1:]))
        if self.output_activation is None:
            self.output_activation = [out]
        else:
            self.output_activation.append(out)
        if len(self.output_activation) == len(self.matrix_names):
            raise ValueError() # we raise an error to stop the forward pass and return the output caught by the hook

    def output_hook_pre(self, module, inp):
        # save the input of the submodule to return it as the output of the forward method
        if isinstance(inp, tuple):
            inp = inp[0]
        if inp.size(0) != self.batch_size:
            inp = inp.reshape([self.batch_size, -1] + list(inp.shape[1:]))
        self.output_activation = (inp,)
        raise ValueError() # we raise an error to stop the forward pass and return the output caught by the hook

    def activate_pass_through(self):
        self.pass_through = True
        self.remove_hook()

    def deactivate_pass_through(self):
        self.pass_through = False

    def forward(self, *args, **kwargs):
        self.batch_size = args[0].shape[0]
        self.register_hook()
        if self.output_hook_handles is not None and len(self.output_hook_handles) > 0:
            try:
                out = super().forward(*args, **kwargs)
            except ValueError:
                pass
            if len(self.output_activation) == 1:
                out = self.output_activation[0]
            else:
                out = torch.cat(self.output_activation, dim=1)
        else:
            out = super().forward(*args, **kwargs)
        self.remove_hook()
        self.output_activation = None
        self.batch_size = None
        return out
    

class MLPWrapper(nn.Module):
    def __init__(self, module: torch.nn.Module, set_submodules: bool = True):
        super().__init__()
        # hide the module from methods like .parameters() by wrapping it in a tuple
        self.module = (module,)
        if set_submodules:
            # we set the submodules of the subblock we want to use as attributes so
            # they are found by methods like .parameters()
            self.set_submodules(module)
    
    def set_submodules(self, module: torch.nn.Module):
        self.final_layer_norm = module.final_layer_norm
        self.fc1 = module.fc1
        self.fc2 = module.fc2

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_head_mask: torch.Tensor | None = None,
        past_key_value: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        position_ids: torch.LongTensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        
        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.module[0].do_layer_norm_before:
            hidden_states = self.module[0].final_layer_norm(hidden_states)

        hidden_states = self.module[0].fc1(hidden_states)
        hidden_states = self.module[0].activation_fn(hidden_states)

        hidden_states = self.module[0].fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.module[0].dropout, training=self.module[0].training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.module[0].do_layer_norm_before:
            hidden_states = self.module[0].final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        return outputs
    

class MLPWrapperLlama(nn.Module):
    def __init__(self, module: torch.nn.Module, set_submodules: bool = True):
        super().__init__()
        self.module = (module,)
        if set_submodules:
            self.set_submodules(module)
    
    def set_submodules(self, module: torch.nn.Module):
        self.mlp = module.mlp
        self.post_attention_layernorm = module.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[transformers.cache_utils.Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: transformers.processing_utils.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.module[0].post_attention_layernorm(hidden_states)
        hidden_states = self.module[0].mlp(hidden_states)
        if residual.device != hidden_states.device:
            hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class MLPWrapperQwen(nn.Module):
    def __init__(self, module: torch.nn.Module, set_submodules: bool = True):
        super().__init__()
        self.module = (module,)
        if set_submodules:
            self.set_submodules(module)

    def set_submodules(self, module: torch.nn.Module):
        self.mlp = module.mlp
        self.post_attention_layernorm = module.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[transformers.cache_utils.Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: transformers.processing_utils.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor]:

        residual = hidden_states
        hidden_states = self.module[0].post_attention_layernorm(hidden_states)
        hidden_states = self.module[0].mlp(hidden_states)
        if residual.device != hidden_states.device:
            hidden_states = hidden_states.to(residual.device)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class MLPWrapperPerMatrix(MLPWrapper):
    def __init__(self, module: torch.nn.Module, matrix_name: str, propagate_to: str | None = None, last_in_block: bool = False,
                 include_ln: bool = False):
        assert matrix_name in ["fc1", "fc2"], "Invalid matrix name."
        self.matrix_name = matrix_name
        self.output_activation = None
        self.output_hook_handle = None
        self.batch_size = None
        super().__init__(module, set_submodules=False)
        self.set_submodule(module, matrix_name)
        if include_ln:
            self.ln = getattr(module, "final_layer_norm")
        self.propagate_to = propagate_to
        self.last_in_block = last_in_block
        self.pass_through = False

    def set_submodule(self, module: torch.nn.Module, matrix_name: str):
        self.matrix = getattr(module, matrix_name)

    def activate_pass_through(self):
        self.pass_through = True
        self.remove_hook()

    def deactivate_pass_through(self):
        self.pass_through = False

    def register_hook(self):
        if not self.pass_through:
            assert self.output_hook_handle is None, "Hook already registered."
            if self.propagate_to is None:
                self.output_hook_handle = self.matrix.register_forward_hook(self.output_hook_post)
            else:
                self.output_hook_handle = getattr(self.module[0], self.propagate_to).register_forward_pre_hook(self.output_hook_pre)

    def remove_hook(self):
        if self.output_hook_handle is not None:
            self.output_hook_handle.remove()
            self.output_hook_handle = None

    def output_hook_post(self, module, inp, out):
        # save the output of the submodule to return it as the output of the forward method
        if isinstance(out, tuple):
            out = out[0]
        if out.size(0) != self.batch_size:
            out = out.reshape([self.batch_size, -1] + list(out.shape[1:]))
        self.output_activation = (out,)
        raise ValueError

    def output_hook_pre(self, module, inp):
        # save the input of the submodule to return it as the output of the forward method
        if isinstance(inp, tuple):
            inp = inp[0]
        if inp.size(0) != self.batch_size:
            inp = inp.reshape([self.batch_size, -1] + list(inp.shape[1:]))
        self.output_activation = (inp,)
        raise ValueError

    def forward(self, *args, **kwargs):
        self.batch_size = args[0].shape[0]
        self.register_hook()
        if self.output_hook_handle is not None:
            try:
                out = super().forward(*args, **kwargs)
            except ValueError:
                pass
            out = self.output_activation[0]
        else:
            out = super().forward(*args, **kwargs)
        self.remove_hook()
        self.output_activation = None
        self.batch_size = None
        return out


class MLPWrapperPerMatrixLlama(MLPWrapperLlama):
    def __init__(self, module: torch.nn.Module, matrix_name: str, last_in_block: bool = False):
        assert matrix_name in ["gate_proj", "up_proj", "down_proj"], "Invalid matrix name."
        self.matrix_name = matrix_name
        self.output_activation = None
        self.output_hook_handle = None
        self.batch_size = None
        super().__init__(module, set_submodules=False)
        self.set_submodule(module, matrix_name)
        self.last_in_block = last_in_block
        self.pass_through = False

    def set_submodule(self, module: torch.nn.Module, matrix_name: str):
        if hasattr(module, matrix_name):
            self.matrix = getattr(module, matrix_name)
        else:
            self.matrix = getattr(module.mlp, matrix_name)

    def activate_pass_through(self):
        self.pass_through = True
        self.remove_hook()

    def deactivate_pass_through(self):
        self.pass_through = False

    def register_hook(self):
        if not self.pass_through:
            assert self.output_hook_handle is None, "Hook already registered."
            self.output_hook_handle = self.matrix.register_forward_hook(self.output_hook_post)

    def remove_hook(self):
        if self.output_hook_handle is not None:
            self.output_hook_handle.remove()
            self.output_hook_handle = None

    def output_hook_post(self, module, inp, out):
        # save the output of the submodule to return it as the output of the forward method
        if isinstance(out, tuple):
            out = out[0]
        if out.size(0) != self.batch_size:
            out = out.reshape([self.batch_size, -1] + list(out.shape[1:]))
        self.output_activation = (out,)
        raise ValueError

    def output_hook_pre(self, module, inp):
        # save the input of the submodule to return it as the output of the forward method
        if isinstance(inp, tuple):
            inp = inp[0]
        if inp.size(0) != self.batch_size:
            inp = inp.reshape([self.batch_size, -1] + list(inp.shape[1:]))
        self.output_activation = (inp,)
        raise ValueError

    def forward(self, *args, **kwargs):
        self.batch_size = args[0].shape[0]
        self.register_hook()
        if self.output_hook_handle is not None:
            try:
                out = super().forward(*args, **kwargs)
            except ValueError:
                pass
            out = self.output_activation[0]
        else:
            out = super().forward(*args, **kwargs)
        self.remove_hook()
        self.output_activation = None
        self.batch_size = None
        return out

    
class DictAccessor(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'DictAccessor' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class RecErrorWrapperLlama(torch.nn.Module):
    '''
    Wrapper for two LLaMA transformer blocks to accumulate the running mean reconstruction error per matrix.
    '''
    def __init__(self, module_original: torch.nn.Module, module_pruned: torch.nn.Module,
                 matrix_name: str | None = None, loss_fn: torch.nn.Module = None):
        super().__init__()
        self.matrix_names = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        if matrix_name is not None:
            assert matrix_name in self.matrix_names, "Invalid matrix name."
            self.matrix_names = [matrix_name]
        self.module_original = module_original
        self.module_pruned = module_pruned
        self.hook_handles = []
        self.accum_errors_mean = {mn: 0 for mn in self.matrix_names}
        self.accum_errors_max = {mn: 0 for mn in self.matrix_names}
        self.output_activations_original = {mn: None for mn in self.matrix_names}
        self.output_activations_pruned = {mn: None for mn in self.matrix_names}
        self.sample_counts = {mn: 0 for mn in self.matrix_names}
        self.reset_accum_errors()
        self.reset_output_activations()
        self.loss_fn = loss_fn

    def reset_accum_errors(self):
        assert isinstance(self.accum_errors_mean, dict), "accum_errors_mean must be a dictionary"
        assert set(self.accum_errors_mean.keys()) == set(self.matrix_names), "Invalid or missing matrix name in accum_errors_mean."
        assert isinstance(self.accum_errors_max, dict), "accum_errors_max must be a dictionary"
        assert set(self.accum_errors_max.keys()) == set(self.matrix_names), "Invalid or missing matrix name in accum_errors_max."
        for matrix_name in self.matrix_names:
            self.accum_errors_mean[matrix_name] = 0
            self.accum_errors_max[matrix_name] = 0
        self.sample_counts = {}
        for matrix_name in self.matrix_names:
            self.sample_counts[matrix_name] = 0

    def reset_output_activations(self):
        assert isinstance(self.output_activations_original, dict), "output_activations_original must be a dictionary"
        assert isinstance(self.output_activations_pruned, dict), "output_activations_pruned must be a dictionary"
        assert set(self.output_activations_original.keys()) == set(self.matrix_names), "Invalid or missing matrix name in output_activations_original."
        assert set(self.output_activations_pruned.keys()) == set(self.matrix_names), "Invalid or missing matrix name in output_activations_pruned."
        for matrix_name in self.matrix_names:
            self.output_activations_original[matrix_name] = None
            self.output_activations_pruned[matrix_name] = None

    def accumulate_error(self):
        for matrix_name in self.matrix_names:
            if self.loss_fn is not None and "cosine" in self.loss_fn:
                error = torch.sum(self.output_activations_original[matrix_name].float() * self.output_activations_pruned[matrix_name].float(), dim=-1)
                error /= torch.sum(self.output_activations_original[matrix_name].float()**2, dim=-1)
                error /= torch.sum(self.output_activations_pruned[matrix_name].float()**2, dim=-1)
            else:
                error = torch.sum((self.output_activations_original[matrix_name].float() - self.output_activations_pruned[matrix_name].float())**2, dim=-1)
            error /= torch.sum(self.output_activations_original[matrix_name].float()**2, dim=-1)
            error_max = torch.max(error)
            error_mean = error.sum().item()
            self.sample_counts[matrix_name] += self.output_activations_original[matrix_name].shape[0]
            self.accum_errors_mean[matrix_name] *= (self.sample_counts[matrix_name] - self.output_activations_original[matrix_name].shape[0]) / self.sample_counts[matrix_name]
            self.accum_errors_mean[matrix_name] += error_mean / self.sample_counts[matrix_name]
            self.accum_errors_max[matrix_name] = max(self.accum_errors_max[matrix_name], error_max)
        self.reset_output_activations()

    def hook(self, matrix_name, original):
        if original:
            def hook_fn(module, inp, out):
                shape = out.shape
                if len(shape) > 2:
                    self.output_activations_original[matrix_name] = out.detach().reshape(-1, shape[-1])
                else:
                    self.output_activations_original[matrix_name] = out.detach()
        else:
            def hook_fn(module, inp, out):
                shape = out.shape
                if len(shape) > 2:
                    self.output_activations_pruned[matrix_name] = out.detach().reshape(-1, shape[-1])
                else:
                    self.output_activations_pruned[matrix_name] = out.detach()
        return hook_fn

    def register_hooks(self):
        assert len(self.hook_handles) == 0, "Hooks already registered."
        for matrix_name in self.matrix_names:
            if matrix_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                self.hook_handles.append(getattr(self.module_original.self_attn, matrix_name).register_forward_hook(self.hook(matrix_name, original=True)))
                self.hook_handles.append(getattr(self.module_pruned.self_attn, matrix_name).register_forward_hook(self.hook(matrix_name, original=False)))
            else:
                self.hook_handles.append(getattr(self.module_original.mlp, matrix_name).register_forward_hook(self.hook(matrix_name, original=True)))
                self.hook_handles.append(getattr(self.module_pruned.mlp, matrix_name).register_forward_hook(self.hook(matrix_name, original=False)))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def free(self):
        self.remove_hooks()
        self.reset_output_activations()
        self.reset_accum_errors()

    def forward(self, x_original, x_pruned, **kwargs):
        x_original = self.module_original(x_original, **kwargs)
        x_pruned = self.module_pruned(x_pruned, **kwargs)

        self.accumulate_error()

        return x_original, x_pruned

        
@torch.no_grad()
def check_reconstruction_error_per_matrix(original_model, pruned_model, args, runner):
    device = args.device
    layers_original = Utils.get_reconstruction_layers(model=original_model, block_size=1, rec=True, runner=runner)
    layers_pruned = Utils.get_reconstruction_layers(model=pruned_model, block_size=1, rec=True, runner=runner)
    
    if args.reconstruct_with_max_information_data:
        sys.stdout.write("Loading calibration data filtered for constant seqlen.\n")
        train_dataloader = Utils.get_c4_for_calibration(nsamples=args.reconstruct_n_samples, seed=args.seed,
                                                        seqlen=original_model.seqlen, tokenized_dataset=runner.get_dataset(args.train_dataset_name),
                                                        tokenizer=runner.tokenizer, split="train")
        validation_dataloader = Utils.get_c4_for_calibration(nsamples=32, seed=1,
                                                        seqlen=original_model.seqlen, tokenized_dataset=runner.get_dataset(args.train_dataset_name),
                                                        tokenizer=runner.tokenizer, split="validation")
    else:
        sys.stdout.write("Loading calibration data without filtering.\n")
        train_dataloader = Utils.get_c4_for_calibration_no_filter(nsamples=args.reconstruct_n_samples, seed=args.seed,
                                                            tokenized_dataset=runner.get_dataset(args.train_dataset_name),
                                                            tokenizer=runner.tokenizer, runner=runner, split="train")
        validation_dataloader = Utils.get_c4_for_calibration_no_filter(nsamples=32, seed=1,
                                                            tokenized_dataset=runner.get_dataset(args.train_dataset_name),
                                                            tokenizer=runner.tokenizer, runner=runner, split="validation")

    inps_tr_orig, outs_tr_orig, attention_mask_tr, position_ids_tr, position_embeddings_tr, loss_mask_tr = PruneUtils.prepare_calibration_input(args, original_model, train_dataloader, device, pad_token_id=runner.tokenizer.pad_token_id,
                                                                                                                         mask_pad_tokens=args.mask_pad_tokens)
    inps_tr_pruned, outs_tr_pruned, _, _, _, _ = PruneUtils.prepare_calibration_input(args, pruned_model, train_dataloader, device, pad_token_id=runner.tokenizer.pad_token_id,
                                                                                                                         mask_pad_tokens=args.mask_pad_tokens)
    inps_val_orig, outs_val_orig, attention_mask_val, position_ids_val, position_embeddings_val, loss_mask_val = PruneUtils.prepare_calibration_input(args, original_model, validation_dataloader, device, pad_token_id=runner.tokenizer.pad_token_id,
                                                                                                                         mask_pad_tokens=args.mask_pad_tokens)
    inps_val_pruned, outs_val_pruned, _, _, _, _ = PruneUtils.prepare_calibration_input(args, pruned_model, validation_dataloader, device, pad_token_id=runner.tokenizer.pad_token_id,
                                                                                                                         mask_pad_tokens=args.mask_pad_tokens)

    errors_train_mean = {}
    errors_val_mean = {}
    errors_train_max = {}
    errors_val_max = {}
    for i, (layer_original, layer_pruned) in enumerate(zip(layers_original, layers_pruned)):
        rec_error_wrapper_tr = RecErrorWrapperLlama(layer_original, layer_pruned, loss_fn=runner.config.loss_fn)
        rec_error_wrapper_tr.register_hooks()
        rec_error_wrapper_val = RecErrorWrapperLlama(layer_original, layer_pruned, loss_fn=runner.config.loss_fn)
        rec_error_wrapper_val.register_hooks()

        zip_list_tr = [
            inps_tr_orig, inps_tr_pruned,
            attention_mask_tr if attention_mask_tr is not None else [None] * len(inps_tr_orig),
            position_ids_tr if position_ids_tr is not None else [None] * len(inps_tr_orig),
            position_embeddings_tr if position_embeddings_tr is not None else [None] * len(inps_tr_orig),
        ]
        zip_list_val = [
            inps_val_orig, inps_val_pruned,
            attention_mask_val if attention_mask_val is not None else [None] * len(inps_val_orig),
            position_ids_val if position_ids_val is not None else [None] * len(inps_val_orig),
            position_embeddings_val if position_embeddings_val is not None else [None] * len(inps_val_orig),
        ]

        for j, (inpo, inpp, amask, pids, pembeds) in enumerate(zip(*zip_list_tr)):
            kwargs = {}
            if amask is not None:
                kwargs['attention_mask'] = amask.to(device)
            if pids is not None:
                kwargs['position_ids'] = pids.to(device)
            if pembeds is not None:
                kwargs['position_embeddings'] = (pembeds[0].to(device), pembeds[1].to(device))
            
            out = rec_error_wrapper_tr(inpo.to(device), inpp.to(device), **kwargs)
            if (new_shape := outs_tr_orig[j].shape) != out[0].shape and outs_tr_orig[j].numel() == out[0].numel():
                out[0] = out[0].reshape(new_shape)
            if (new_shape := outs_tr_pruned[j].shape) != out[1].shape and outs_tr_pruned[j].numel() == out[1].numel():
                out[1] = out[1].reshape(new_shape)
            outs_tr_orig[j] = out[0].cpu()
            outs_tr_pruned[j] = out[1].cpu()

        for j, (inpo, inpp, amask, pids, pembeds) in enumerate(zip(*zip_list_val)):
            kwargs = {}
            if amask is not None:
                kwargs['attention_mask'] = amask.to(device)
            if pids is not None:
                kwargs['position_ids'] = pids.to(device)
            if pembeds is not None:
                kwargs['position_embeddings'] = (pembeds[0].to(device), pembeds[1].to(device))
            out = rec_error_wrapper_val(inpo.to(device), inpp.to(device), **kwargs)
            if (new_shape := outs_val_orig[j].shape) != out[0].shape and outs_val_orig[j].numel() == out[0].numel():
                out[0] = out[0].reshape(new_shape)
            if (new_shape := outs_val_pruned[j].shape) != out[1].shape and outs_val_pruned[j].numel() == out[1].numel():
                out[1] = out[1].reshape(new_shape)
            outs_val_orig[j] = out[0].cpu()
            outs_val_pruned[j] = out[1].cpu()

        for mn in rec_error_wrapper_tr.matrix_names:
            errors_train_mean[f"cal_set_layer_{i+1}_{mn}"] = rec_error_wrapper_tr.accum_errors_mean[mn]
            errors_train_max[f"cal_set_layer_{i+1}_{mn}"] = rec_error_wrapper_tr.accum_errors_max[mn]
            errors_val_mean[f"val_set_layer_{i+1}_{mn}"] = rec_error_wrapper_val.accum_errors_mean[mn]
            errors_val_max[f"val_set_layer_{i+1}_{mn}"] = rec_error_wrapper_val.accum_errors_max[mn]
        
        inps_tr_orig = [o.clone() for o in outs_tr_orig]
        inps_tr_pruned = [o.clone() for o in outs_tr_pruned]
        inps_val_orig = [o.clone() for o in outs_val_orig]
        inps_val_pruned = [o.clone() for o in outs_val_pruned]
    
        rec_error_wrapper_tr.free()
        rec_error_wrapper_val.free()
    return errors_train_mean, errors_val_mean, errors_train_max, errors_val_max


@torch.no_grad()
def check_local_reconstruction_error_per_matrix(original_model, pruned_model, args, runner):
    device = args.device
    layers_original = Utils.get_reconstruction_layers(model=original_model, block_size=1, rec=True, runner=runner)
    layers_pruned = Utils.get_reconstruction_layers(model=pruned_model, block_size=1, rec=True, runner=runner)
    matrix_names = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    
    if args.reconstruct_with_max_information_data:
        sys.stdout.write("Loading calibration data filtered for constant seqlen.\n")
        train_dataloader = Utils.get_c4_for_calibration(nsamples=args.reconstruct_n_samples, seed=args.seed,
                                                        seqlen=original_model.seqlen, tokenized_dataset=runner.get_dataset(args.train_dataset_name),
                                                        tokenizer=runner.tokenizer, split="train")
        validation_dataloader = Utils.get_c4_for_calibration(nsamples=32, seed=1,
                                                        seqlen=original_model.seqlen, tokenized_dataset=runner.get_dataset(args.train_dataset_name),
                                                        tokenizer=runner.tokenizer, split="validation")
    else:
        sys.stdout.write("Loading calibration data without filtering.\n")
        train_dataloader = Utils.get_c4_for_calibration_no_filter(nsamples=args.reconstruct_n_samples, seed=args.seed,
                                                            tokenized_dataset=runner.get_dataset(args.train_dataset_name),
                                                            tokenizer=runner.tokenizer, runner=runner, split="train")
        validation_dataloader = Utils.get_c4_for_calibration_no_filter(nsamples=32, seed=1,
                                                            tokenized_dataset=runner.get_dataset(args.train_dataset_name),
                                                            tokenizer=runner.tokenizer, runner=runner, split="validation")

    inps_tr_orig, outs_tr_orig, attention_mask_tr, position_ids_tr, position_embeddings_tr, loss_mask_tr = PruneUtils.prepare_calibration_input(args, original_model, train_dataloader, device, pad_token_id=runner.tokenizer.pad_token_id,
                                                                                                                         mask_pad_tokens=args.mask_pad_tokens)
    _, outs_tr_pruned, _, _, _, _ = PruneUtils.prepare_calibration_input(args, pruned_model, train_dataloader, device, pad_token_id=runner.tokenizer.pad_token_id,
                                                                                                                         mask_pad_tokens=args.mask_pad_tokens)
    inps_val_orig, outs_val_orig, attention_mask_val, position_ids_val, position_embeddings_val, loss_mask_val = PruneUtils.prepare_calibration_input(args, original_model, validation_dataloader, device, pad_token_id=runner.tokenizer.pad_token_id,
                                                                                                                         mask_pad_tokens=args.mask_pad_tokens)
    _, outs_val_pruned, _, _, _, _ = PruneUtils.prepare_calibration_input(args, pruned_model, validation_dataloader, device, pad_token_id=runner.tokenizer.pad_token_id,
                                                                                                                         mask_pad_tokens=args.mask_pad_tokens)

    errors_train_mean = {}
    errors_val_mean = {}
    errors_train_max = {}
    errors_val_max = {}

    def get_matrix(layer, matrix_name):
        if matrix_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            return getattr(layer.self_attn, matrix_name)
        else:
            return getattr(layer.mlp, matrix_name)


    for i, (layer_original, layer_pruned) in enumerate(zip(layers_original, layers_pruned)):
        # for every matrix, check the error after that matrix as if it is the only pruned matrix in the model
        pruned_matrices = {matrix_name: get_matrix(layer_pruned, matrix_name).weight.data.clone() for matrix_name in matrix_names}
        for matrix_name in matrix_names:
            for matrix_name2 in matrix_names: # reset all matrices to original
                get_matrix(layer_pruned, matrix_name2).weight.data = get_matrix(layer_original, matrix_name2).weight.data.clone()
            # set the current matrix to the pruned matrix
            get_matrix(layer_pruned, matrix_name).weight.data = pruned_matrices[matrix_name].clone()

            rec_error_wrapper_tr = RecErrorWrapperLlama(layer_original, layer_pruned, matrix_name=matrix_name, loss_fn=runner.config.loss_fn)
            rec_error_wrapper_tr.register_hooks()
            rec_error_wrapper_val = RecErrorWrapperLlama(layer_original, layer_pruned, matrix_name=matrix_name, loss_fn=runner.config.loss_fn)
            rec_error_wrapper_val.register_hooks()

            zip_list_tr = [
                inps_tr_orig,
                attention_mask_tr if attention_mask_tr is not None else [None] * len(inps_tr_orig),
                position_ids_tr if position_ids_tr is not None else [None] * len(inps_tr_orig),
                position_embeddings_tr if position_embeddings_tr is not None else [None] * len(inps_tr_orig),
            ]
            zip_list_val = [
                inps_val_orig,
                attention_mask_val if attention_mask_val is not None else [None] * len(inps_val_orig),
                position_ids_val if position_ids_val is not None else [None] * len(inps_val_orig),
                position_embeddings_val if position_embeddings_val is not None else [None] * len(inps_val_orig),
            ]

            for j, (inpo, amask, pids, pembeds) in enumerate(zip(*zip_list_tr)):
                kwargs = {}
                if amask is not None:
                    kwargs['attention_mask'] = amask.to(device)
                if pids is not None:
                    kwargs['position_ids'] = pids.to(device)
                if pembeds is not None:
                    kwargs['position_embeddings'] = (pembeds[0].to(device), pembeds[1].to(device))
                
                out = rec_error_wrapper_tr(inpo.to(device), inpo.to(device), **kwargs)
                if (new_shape := outs_tr_orig[j].shape) != out[0].shape and outs_tr_orig[j].numel() == out[0].numel():
                    out[0] = out[0].reshape(new_shape)
                if (new_shape := outs_tr_pruned[j].shape) != out[1].shape and outs_tr_pruned[j].numel() == out[1].numel():
                    out[1] = out[1].reshape(new_shape)
                outs_tr_orig[j] = out[0].cpu()
                outs_tr_pruned[j] = out[1].cpu()

            for j, (inpo, amask, pids, pembeds) in enumerate(zip(*zip_list_val)):
                kwargs = {}
                if amask is not None:
                    kwargs['attention_mask'] = amask.to(device)
                if pids is not None:
                    kwargs['position_ids'] = pids.to(device)
                if pembeds is not None:
                    kwargs['position_embeddings'] = (pembeds[0].to(device), pembeds[1].to(device))
                out = rec_error_wrapper_val(inpo.to(device), inpo.to(device), **kwargs)
                if (new_shape := outs_val_orig[j].shape) != out[0].shape and outs_val_orig[j].numel() == out[0].numel():
                    out[0] = out[0].reshape(new_shape)
                if (new_shape := outs_val_pruned[j].shape) != out[1].shape and outs_val_pruned[j].numel() == out[1].numel():
                    out[1] = out[1].reshape(new_shape)
                outs_val_orig[j] = out[0].cpu()
                outs_val_pruned[j] = out[1].cpu()

            for mn in rec_error_wrapper_tr.matrix_names:
                errors_train_mean[f"cal_set_layer_{i+1}_{mn}"] = rec_error_wrapper_tr.accum_errors_mean[mn]
                errors_train_max[f"cal_set_layer_{i+1}_{mn}"] = rec_error_wrapper_tr.accum_errors_max[mn]
                errors_val_mean[f"val_set_layer_{i+1}_{mn}"] = rec_error_wrapper_val.accum_errors_mean[mn]
                errors_val_max[f"val_set_layer_{i+1}_{mn}"] = rec_error_wrapper_val.accum_errors_max[mn]

            rec_error_wrapper_tr.free()
            rec_error_wrapper_val.free()
            
        inps_tr_orig = [o.clone() for o in outs_tr_orig]
        inps_val_orig = [o.clone() for o in outs_val_orig]

    return errors_train_mean, errors_val_mean, errors_train_max, errors_val_max


def get_block_importance(args, layers, inputs, outs, attention_mask, position_ids, position_embeddings, loss_mask, device):
    """
    Compute importances as cosine similarities of inputs and outputs of every layer.
    """
    importances = []
    for layer in layers:
        Utils.get_outputs(args=args, layer=layer, inps=inputs, outs=outs, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings, loss_mask=loss_mask, device=device)
        x = 0
        for inp, out in zip(inputs, outs):
            x += torch.cosine_similarity(inp.view(-1, inp.shape[-1]), out.view(-1, out.shape[-1]), dim=1).mean().item()
        importances.append(x)
        inputs = [o.clone() for o in outs]
    return importances

# functions for FLAP

def infer_attn_dims(attn):
    cfg = getattr(attn, "config", None)

    n_heads = cfg.num_attention_heads if (cfg is not None and hasattr(cfg, "num_attention_heads")) else None
    n_kv = cfg.num_key_value_heads if (cfg is not None and hasattr(cfg, "num_key_value_heads")) else None

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and cfg is not None:
        head_dim = getattr(cfg, "head_dim", None)

    q_out = attn.q_proj.weight.shape[0]
    if head_dim is None:
        if n_heads is None:
            raise ValueError("Can't infer head_dim (need config.num_attention_heads or head_dim).")
        head_dim = q_out // n_heads

    if n_heads is None:
        n_heads = q_out // head_dim

    if n_kv is None:
        k_out = attn.k_proj.weight.shape[0]
        n_kv = k_out // head_dim

    return n_heads, n_kv, head_dim


def build_q_kv_channel_masks(attn_mask, n_heads, n_kv, head_dim):
    attn_mask = attn_mask.to(dtype=torch.bool)

    q_chan_mask = attn_mask.repeat_interleave(head_dim)  # [n_heads * head_dim]

    if n_kv == n_heads:
        kv_head_mask = attn_mask
    else:
        group = n_heads // n_kv
        kv_head_mask = attn_mask.view(n_kv, group).any(dim=1)

    kv_chan_mask = kv_head_mask.repeat_interleave(head_dim)  # [n_kv * head_dim]
    return q_chan_mask, kv_chan_mask


def mask_linear_rows(linear, chan_mask, device):
    # mask rows (out_features)
    mask = chan_mask[:, None].expand(linear.weight.shape[0], linear.weight.shape[1])
    prune.custom_from_mask(linear, name="weight", mask=mask.to(device))


def mask_linear_cols(linear, chan_mask, device):
    # mask cols (in_features)
    mask = chan_mask[None, :].expand(linear.weight.shape[0], linear.weight.shape[1])
    prune.custom_from_mask(linear, name="weight", mask=mask.to(device))


def safe_set_bias(linear, new_bias):
    if getattr(linear, "bias", None) is not None:
        linear.bias.data = new_bias


def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    """
    This function is adapted from the original compress function in the prune.py file to work
    with GQA.
    (Original code: https://github.com/CASIA-LMC-Lab/FLAP)

    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """

    if not unstr:
        raise NotImplementedError("This rewrite covers the unstr path only.")
    
    if attn_mask is not None:
        attn = layer.self_attn
        n_heads, n_kv, head_dim = infer_attn_dims(attn)

        # attn_mask is expected to be per-query-head: shape [n_heads]
        if attn_mask.numel() != n_heads:
            raise ValueError(f"attn_mask has {attn_mask.numel()} elems, expected num_heads={n_heads}.")

        q_chan_mask, kv_chan_mask = build_q_kv_channel_masks(attn_mask, n_heads, n_kv, head_dim)

        # q/k/v projections: mask rows (out_features)
        mask_linear_rows(attn.q_proj, q_chan_mask, device)
        mask_linear_rows(attn.k_proj, kv_chan_mask, device)
        mask_linear_rows(attn.v_proj, kv_chan_mask, device)

        # o projection: mask cols (in_features = n_heads*head_dim)
        mask_linear_cols(attn.o_proj, q_chan_mask, device)

        # Bias compensation for o_proj only if it actually has a bias (LLaMA may; Qwen2 often doesn't)
        if bias and attn_mean_inp is not None and getattr(attn.o_proj, "bias", None) is not None:
            o_w = attn.o_proj.weight.data  # [hidden, n_heads*head_dim]
            dropped = (~q_chan_mask.to(device))

            # Support either [n_heads*head_dim] or [B, n_heads*head_dim] mean input
            if attn_mean_inp.dim() == 1:
                output_bias = (attn_mean_inp.to(device) * dropped) @ o_w.T  # [hidden]
            else:
                # If you stored a batch of means, average them first
                mean_vec = attn_mean_inp.to(device).mean(dim=0)
                output_bias = (mean_vec * dropped) @ o_w.T  # [hidden]

            safe_set_bias(attn.o_proj, output_bias)

    if mlp_mask is not None:
        mlp_mask = mlp_mask.to(dtype=torch.bool)

        # up/gate: mask rows (out_features)
        mask_linear_rows(layer.mlp.up_proj, mlp_mask, device)
        mask_linear_rows(layer.mlp.gate_proj, mlp_mask, device)

        # down: mask cols (in_features)
        mask_linear_cols(layer.mlp.down_proj, mlp_mask, device)

        # Bias compensation for down_proj only if it has a bias
        if bias and mlp_mean_inp is not None and getattr(layer.mlp.down_proj, "bias", None) is not None:
            d_w = layer.mlp.down_proj.weight.data  # [hidden, intermediate]
            dropped = (~mlp_mask.to(device))

            if mlp_mean_inp.dim() == 1:
                output_bias = (mlp_mean_inp.to(device) * dropped) @ d_w.T  # [hidden]
            else:
                mean_vec = mlp_mean_inp.to(device).mean(dim=0)
                output_bias = (mean_vec * dropped) @ d_w.T  # [hidden]

            safe_set_bias(layer.mlp.down_proj, output_bias)
        
    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()