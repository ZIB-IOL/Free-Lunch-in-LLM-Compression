import sys
from tqdm import tqdm
from typing import NamedTuple, Optional 
import torch
import torch.nn as nn

import torch.nn.utils.prune as prune
from utilities import Utils, PruneUtils, WandaWrapper, SparseGPTWrapper, SelectiveMethods, AttnWrapper, MLPWrapper, FLAPWrapper, compress
from torch.optim import AdamW, SGD
from transformers.optimization import get_linear_schedule_with_warmup
import peft_methods

class PruneMethod:

    def __init__(self, runner, args: NamedTuple, prune_method: str, model: torch.nn.Module, prune_n: int = 0, prune_m: int = 0):
        self.runner = runner
        self.args = args
        self.prune_method = prune_method
        self.model = model
        self.prune_n = prune_n
        self.prune_m = prune_m       

        assert self.prune_method in ["magnitude", "random", "wanda", "sparsegpt", "ria", "wanda_sp", "magnitude_sp", "flap"], "Invalid pruning method."
        self.requires_calibration_wrapper = self.prune_method in ["wanda", "sparsegpt", "ria", "wanda_sp", "flap"]

        self.reconstruction_error_initial, self.reconstruction_error_final = None, None

    @torch.no_grad()
    def prune(self, do_prune: bool = True):
        device = self.args.device

        # split decoder into submodels
        reconstruction_block_size = self.args.block_size
        pruning_block_size = self.args.pruning_block_size or reconstruction_block_size
        layers_rec = Utils.get_reconstruction_layers(model=self.model, block_size=reconstruction_block_size, rec=True, runner=self.runner)
        layers_prune = Utils.get_reconstruction_layers(model=self.model, block_size=pruning_block_size, rec=False, runner=self.runner)

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        train_losses = {}
        val_losses = {}
        grad_norms = {}

        # get calibration data
        if self.args.reconstruct or self.requires_calibration_wrapper:
            if self.args.reconstruct_with_max_information_data:
                sys.stdout.write("Loading calibration data filtered for constant seqlen.\n")
                dataloader = Utils.get_c4_for_calibration(nsamples=self.args.reconstruct_n_samples, seed=self.args.seed,
                                                          seqlen=self.model.seqlen, tokenized_dataset=self.runner.get_dataset(self.args.train_dataset_name),
                                                          tokenizer=self.runner.tokenizer)
            else:
                sys.stdout.write("Loading calibration data without filtering.\n")
                dataloader = Utils.get_c4_for_calibration_no_filter(nsamples=self.args.reconstruct_n_samples, seed=self.args.seed,
                                                                    tokenized_dataset=self.runner.get_dataset(self.args.train_dataset_name),
                                                                    tokenizer=self.runner.tokenizer, runner=self.runner)
            if self.args.log_train_loss:
                if self.args.reconstruct_with_max_information_data:
                    val_loader = Utils.get_c4_for_calibration(nsamples=32, seed=42,
                                                 seqlen=self.model.seqlen, tokenized_dataset=self.runner.get_dataset(self.args.train_dataset_name),
                                                 tokenizer=self.runner.tokenizer, split="validation")
                else:
                    val_loader = Utils.get_c4_for_calibration_no_filter(nsamples=32, seed=42,
                                                                    tokenized_dataset=self.runner.get_dataset(self.args.train_dataset_name),
                                                                    tokenizer=self.runner.tokenizer, runner=self.runner, split="validation")
            else:
                val_loader = None
                                        
        # embed calibration data
        if self.args.reconstruct:
            with torch.no_grad():
                inps_rec, outs_rec, attention_mask, position_ids, position_embeddings, loss_mask =\
                    PruneUtils.prepare_calibration_input(self.args, self.model, dataloader, device, pad_token_id=self.runner.tokenizer.pad_token_id,
                                                         mask_pad_tokens=self.args.mask_pad_tokens)
            if self.args.use_dense_targets:
                dense_inps_rec, dense_outs_rec = [inp.clone() for inp in inps_rec], [out.clone() for out in outs_rec]
            else:
                dense_inps_rec, dense_outs_rec = None, None
            if val_loader is not None:
                inps_val, outs_val, attention_mask_val, position_ids_val, position_embeddings_val, loss_mask_val =\
                    PruneUtils.prepare_calibration_input(self.args, self.model, val_loader, device, pad_token_id=self.runner.tokenizer.pad_token_id,
                                                         mask_pad_tokens=self.args.mask_pad_tokens)
            else:
                inps_val, outs_val, attention_mask_val, position_ids_val, position_embeddings_val, loss_mask_val = None, None, None, None, None, None
            if self.args.use_dense_targets and val_loader is not None:
                dense_inps_val, dense_outs_val = [inp.clone() for inp in inps_val], [out.clone() for out in outs_val]
            else:
                dense_inps_val, dense_outs_val = None, None
        if self.requires_calibration_wrapper:
            if self.args.reconstruct:
                inps_prune, outs_prune = [inp.clone() for inp in inps_rec], [out.clone() for out in outs_rec]
            else:
                with torch.no_grad():
                    inps_prune, outs_prune, attention_mask, position_ids, position_embeddings, loss_mask =\
                        PruneUtils.prepare_calibration_input(self.args, self.model, dataloader, device, pad_token_id=self.runner.tokenizer.pad_token_id,
                                                             mask_pad_tokens=self.args.mask_pad_tokens)

        wrappers, hooks = {}, []

        # figure out how often we should prune and reconstruct
        def get_iterations(x, y, pruning: bool = False):
            if x <= y:
                return 1
            subblockmap = {-1: 2, -2: 6 if self.runner.is_opt else 7} # OPT only has 6 matrices per block
            if x > 0 and y > 0:
                return x // y
            elif x > 0 and y < 0:
                return subblockmap[y] * x
            elif x < 0 and y < 0:
                return subblockmap[y] // subblockmap[x]
            else:
                raise ValueError(f"Invalid block sizes: {x} and {y}")

        prune_every = get_iterations(pruning_block_size, reconstruction_block_size, pruning=True)
        reconstruct_every = get_iterations(reconstruction_block_size, pruning_block_size, pruning=False)
        sys.stdout.write(f"prune_every: {prune_every}, reconstruct_every: {reconstruct_every}.\n")
        iterations_since_last_reconstruction = 0
        iterations_since_last_pruning = 0
        layers_rec_iterator = iter(enumerate(layers_rec))
        layers_prune_iterator = iter(enumerate(layers_prune))
        i_rec, i_prune = 0, 0

        for i in range(max(len(layers_rec), len(layers_prune))):
            new_prune_block, new_reconstruct_block = False, False
            if iterations_since_last_reconstruction == 0 and (i_rec < len(layers_rec) - 1 or (len(layers_rec) == 1 and i_rec == 0)):
                new_reconstruct_block = True
                i_rec, layer_rec = next(layers_rec_iterator)
            if iterations_since_last_pruning == 0 and (i_prune < len(layers_prune) - 1 or (len(layers_prune) == 1 and i_prune == 0)):
                new_prune_block = True
                i_prune, layer_prune = next(layers_prune_iterator)
            iterations_since_last_pruning += 1
            iterations_since_last_reconstruction += 1
            reconstruct_now = iterations_since_last_reconstruction == reconstruct_every
            prune_now = (iterations_since_last_pruning == prune_every or i == 0) and do_prune

            if new_prune_block and do_prune:
                # list of linear sublayers in the current layer
                subset = Utils.get_layers_of_modules(layer_prune)
                pruned_subset = []
                if self.requires_calibration_wrapper:
                    # handle multiple GPUs
                    if hasattr(self.model, "hf_device_map") and f"self.model.layers.{i_prune}" in self.model.hf_device_map:
                        device = self.model.hf_device_map[f"self.model.layers.{i_prune}"]

                    wrappers, hooks = self.get_wrappers_and_hooks(layer_subset=subset)
                    # Changes the outs dynamically
                    Utils.get_outputs(args=self.args, layer=layer_prune, inps=inps_prune, outs=outs_prune,
                                        attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings,
                                        loss_mask=loss_mask, device=device)
                    for h in hooks:
                        h.remove()

            
            if new_reconstruct_block and self.args.reconstruct:
                if not prune_now and do_prune:
                    # handle multiple GPUs
                    if hasattr(self.model, "hf_device_map") and f"self.model.layers.{i_rec}" in self.model.hf_device_map:
                        device = self.model.hf_device_map[f"self.model.layers.{i_rec}"]
                # Changes the outs dynamically
                Utils.get_outputs(args=self.args, layer=layer_rec, inps=inps_rec, outs=outs_rec, attention_mask=attention_mask,
                                  position_ids=position_ids, position_embeddings=position_embeddings, loss_mask=loss_mask, device=device)
                if self.args.use_dense_targets:
                    # Change the outs dynamically
                    Utils.get_outputs(args=self.args, layer=layer_rec, inps=dense_inps_rec, outs=dense_outs_rec, attention_mask=attention_mask,
                                      position_ids=position_ids, position_embeddings=position_embeddings, loss_mask=loss_mask, device=device)
                if val_loader is not None:
                    # Change the outs dynamically
                    Utils.get_outputs(args=self.args, layer=layer_rec, inps=inps_val, outs=outs_val, attention_mask=attention_mask_val,
                                      position_ids=position_ids_val, position_embeddings=position_embeddings_val, loss_mask=loss_mask_val, device=device)
                    if self.args.use_dense_targets:
                        Utils.get_outputs(args=self.args, layer=layer_rec, inps=dense_inps_val, outs=dense_outs_val, attention_mask=attention_mask_val,
                                          position_ids=position_ids_val, position_embeddings=position_embeddings_val, loss_mask=loss_mask_val, device=device)
            
            # prune the block
            if prune_now or (do_prune and pruning_block_size > reconstruction_block_size):
                if pruning_block_size > reconstruction_block_size:
                    # when we reconstruct with a finer granularity than we prune, we save the information for pruning
                    # the whole pruning block but only prune the weights corresponding to the current reconstruction block.
                    # this way we get the correct reconstruction targets when propagating the sparse activations while
                    # still having the pruning metrics as if we were pruning the whole pruning block at once.
                    # in this case, we set iterations_since_last_pruning = 0 after every subsubset has been pruned
                    subset = Utils.get_layers_of_modules(layer_prune)
                    subset_rec = Utils.get_layers_of_modules(layer_rec)
                    keys_to_prune = []
                    for prune_key in subset.keys():
                        for rec_key in subset_rec.keys():
                            if torch.equal(subset[prune_key].weight.data, subset_rec[rec_key].weight.data):
                                keys_to_prune.append(prune_key)
                    pruned_subset.extend(keys_to_prune)
                    if len(pruned_subset) == len(subset):
                        iterations_since_last_pruning = 0
                    subset = {key: subset[key] for key in keys_to_prune}
                    sys.stdout.write(f"Pruning part of layer {i_prune+1} of {len(layers_prune)}\n")
                else:
                    iterations_since_last_pruning = 0
                    sys.stdout.write(f"Pruning layer {i_prune+1} of {len(layers_prune)}\n")

                for name in subset:
                    W = subset[name].weight.data
                    # Determine Saliency criterion
                    if self.prune_method == 'magnitude':
                        # Use the magnitude as saliency
                        W_metric = torch.abs(W)      
                    elif self.prune_method == 'random':
                        # Random saliency
                        W_metric = torch.randn_like(W)
                    elif self.prune_method == 'wanda':
                        # Wanda pruning criterion
                        W_metric = torch.abs(W) * torch.sqrt(wrappers[name].scaler_row.reshape((1,-1)))
                    elif self.prune_method == 'wanda_sp':
                        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrappers[name].scaler_row.reshape((1,-1)))
                        if name == 'self_attn.o_proj':
                            num_heads = self.model.config.num_attention_heads
                            W_metric = W_metric.mean(axis=0).reshape(-1, 128).sum(dim=1)    # importance score of each head
                            thresh = torch.sort(W_metric.cuda())[0][int(self.args.sparsity_ratio*num_heads)].cpu()
                            W_mask = (W_metric>=thresh)
                            compress(layer_prune, W_mask, None, None, None, device, bias=False, unstr=True)
                        elif 'down_proj' in name:
                            W_metric = W_metric.mean(axis=0)
                            thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*self.args.sparsity_ratio)].cpu()
                            W_mask = (W_metric>=thresh)
                            compress(layer_prune, None, W_mask, None, None, device, bias=False, unstr=True)
                        W_metric = None
                        W_mask = None
                    elif self.prune_method == 'ria':
                        # RIA pruning criterion
                        W_metric = (torch.abs(W)/torch.sum(torch.abs(W), dim=0) + torch.abs(W)/torch.sum(torch.abs(W), dim=1).reshape(-1, 1))\
                                * (torch.sqrt(wrappers[name].scaler_row.reshape((1,-1))))**self.args.ria_alpha
                    elif self.prune_method == 'sparsegpt':
                        # SparseGPT pruning criterion, the weights are changed automatically and we infer the mask from the weights, hence the saliency is None
                        wrappers[name].fasterprune(sparsity=self.args.sparsity_ratio, prune_n=self.prune_n, prune_m=self.prune_m, percdamp=0.01, blocksize=self.args.reconstruct_n_samples)
                        wrappers[name].free()                
                        W_metric = None
                        W_mask = (subset[name].weight == 0)

                    # Prune
                    if W_metric is not None:    # True for all methods except wanda_sp and sparsegpt
                        if self.prune_n != 0:
                            W_mask = PruneUtils.get_n_m_pruning_mask(W_saliency=W_metric, prune_n=self.prune_n, prune_m=self.prune_m)
                        elif self.runner.config.sparsity_type == "blockwise":
                            W_mask = PruneUtils.get_block_wise_pruning_mask(W_saliency=W_metric)
                        else:
                            if not self.args.prune_whole_matrix and self.prune_method in ["wanda", "ria"]:
                                sort_res = torch.sort(W_metric, dim=1, stable=True)[1]
                                indices = sort_res[:,:int(W_metric.shape[1]*self.args.sparsity_ratio)]
                                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                                W_mask.scatter_(1, indices, True)
                            else:
                                thresh = torch.sort(W_metric.flatten())[0][int(W.numel()*self.args.sparsity_ratio)]              
                                W_mask = (W_metric <= thresh)
                    # Prune the weights
                    if W_mask is not None:
                        prune.custom_from_mask(subset[name], name='weight', mask=~W_mask)

                    if not self.args.keep_masks and not self.args.reconstruct:
                        # We can remove the masks now already, since we either reconstruct non-pruned parameters or we handle this in the peft method
                        if prune.is_pruned(subset[name]):
                            prune.remove(subset[name], name='weight')
            
            train_loss = None
            val_loss = None
            grad_norm = None
            if reconstruct_now:
                # reconstruct pruned submodel
                iterations_since_last_reconstruction = 0
                if self.args.reconstruct:
                    target = outs_rec if not self.args.use_dense_targets else dense_outs_rec
                    if val_loader is not None:
                        val_args = (inps_val, outs_val, attention_mask_val, position_ids_val, position_embeddings_val, loss_mask_val)
                    else:
                        val_args = None
                    with torch.enable_grad():
                        train_loss, val_loss, grad_norm = self.reconstruct_weights(layer=layer_rec, inps=inps_rec, outs=target, device=device, attention_mask=attention_mask,
                                                    position_ids=position_ids, position_embeddings=position_embeddings, layer_idx=i_rec, n_layers=len(layers_rec),
                                                    loss_mask=loss_mask, keep_masks=self.args.keep_masks, val_args=val_args)
            if train_loss is not None:
                train_losses[f"train_loss_{i_rec+1}"] = train_loss
            if val_loss is not None:
                val_losses[f"val_loss_{i_rec+1}"] = val_loss
            if grad_norm is not None:
                grad_norms[f"grad_norm_{i_rec+1}"] = grad_norm

            # If we do not retrain further, we can now definitely remove the masks
            if (not self.args.keep_masks) and (reconstruct_now if self.args.reconstruct else True) and (not self.args.training_mode == 'retrain'):
                subset = Utils.get_layers_of_modules(layer_rec if self.args.reconstruct else layer_prune)
                for name in subset:
                    if prune.is_pruned(subset[name]):
                        prune.remove(subset[name], name='weight')

            # propagate calibration data for pruning
            if iterations_since_last_pruning == 0 and self.requires_calibration_wrapper and (layer_prune.last_in_block if hasattr(layer_prune, "last_in_block") else True):
                if hasattr(layer_prune, "pass_through"):
                    layer_prune.activate_pass_through()
                if self.args.propagate_sparse_activations_prune:
                    Utils.get_outputs(args=self.args, layer=layer_prune, inps=inps_prune, outs=outs_prune, attention_mask=attention_mask,
                                      position_ids=position_ids, position_embeddings=position_embeddings, loss_mask=loss_mask, device=device)  # Changes the outs dynamically
                inps_prune, outs_prune = outs_prune, inps_prune
                if hasattr(layer_prune, "pass_through"):
                    layer_prune.deactivate_pass_through()

            # propagate calibration data for reconstruction
            if reconstruct_now and self.args.reconstruct and (layer_rec.last_in_block if hasattr(layer_rec, "last_in_block") else True):
                if hasattr(layer_rec, "pass_through"):
                    layer_rec.activate_pass_through()
                if self.args.propagate_sparse_activations_reconstruct:
                    Utils.get_outputs(args=self.args, layer=layer_rec, inps=inps_rec, outs=outs_rec, attention_mask=attention_mask,
                                      position_ids=position_ids, position_embeddings=position_embeddings, loss_mask=loss_mask, device=device)  # Changes the outs dynamically
                inps_rec, outs_rec = outs_rec, inps_rec
                if self.args.use_dense_targets:
                    dense_inps_rec, dense_outs_rec = dense_outs_rec, dense_inps_rec
                if val_loader is not None:
                    if self.args.propagate_sparse_activations_reconstruct:
                        Utils.get_outputs(args=self.args, layer=layer_rec, inps=inps_val, outs=outs_val, attention_mask=attention_mask_val,
                                          position_ids=position_ids_val, position_embeddings=position_embeddings_val, loss_mask=loss_mask_val, device=device)
                    inps_val, outs_val = outs_val, inps_val
                    if self.args.use_dense_targets:
                        dense_inps_val, dense_outs_val = dense_outs_val, dense_inps_val
                if hasattr(layer_rec, "pass_through"):
                    layer_rec.deactivate_pass_through()

            torch.cuda.empty_cache()

        self.model.config.use_cache = use_cache

        # data for logging
        if self.args.log_train_loss:
            self.train_losses = train_losses
            if val_args is not None:
                self.val_losses = val_losses
            else:
                self.val_losses = None
        else:
            self.train_losses = None
            self.val_losses = None
        if self.args.log_grad_norm:
            self.grad_norms = grad_norms
        else:
            self.grad_norms = None

        torch.cuda.empty_cache()

    def reconstruct_weights(self, layer: torch.nn.Module, inps: torch.Tensor, outs: torch.Tensor, device: torch.device, attention_mask: Optional[torch.Tensor],
                            position_ids: Optional[torch.Tensor], position_embeddings: Optional[torch.Tensor], layer_idx: int, n_layers: int,
                            ignore_reconstruction_method: bool = False, loss_mask: Optional[torch.Tensor] = None, keep_masks: bool = True,
                            val_args: Optional[tuple[torch.Tensor, ...]] = None) -> float:
        """Reconstructs the weights of the layer using the input-output pairs."""
        # "dataloader" for the reconstruction
        zip_list = [
            inps,
            outs,
            attention_mask if attention_mask is not None else [None] * len(inps),
            position_ids if position_ids is not None else [None] * len(inps),
            position_embeddings if position_embeddings is not None else [None] * len(inps),
            loss_mask
        ]
        len_tensor_dataloader = len(zip_list[0])

        # enable PEFT or all gradients
        peft_strategy = self.runner.config.peft_strategy
        # enable grad for all parameters that correspond to the peft strategy at stake
        assert hasattr(peft_methods, peft_strategy), f"PEFT strategy {peft_strategy} not implemented."
        peft_strategy = getattr(peft_methods, peft_strategy)(model=self.model, runner=self, config=self.runner.config, total_iterations=self.args.n_iterations, is_reconstruct=True)
        peft_strategy.select_peft_layers(layer=layer)
        # when splitting att and MLP, we need to turn off the gradients for the whole module first,
        # as the layer will only show the attn or MLP parameters to the peft strategy
        if isinstance(layer, AttnWrapper) or isinstance(layer, MLPWrapper):
            SelectiveMethods.deactivate_model(self.model)
            SelectiveMethods.activate_model(layer)
        if self.args.constant_layer_norm:
            SelectiveMethods.deactivate_layer_norm_params(layer)

        # optimizer and scheduler
        lr = float(self.args.initial_lr)
        original_dtype = next(iter(layer.parameters())).dtype
        for param in layer.parameters():
            if param.requires_grad:
                # Important: Set trainable parameters to float32, otherwise this won't work with fp16=True -> https://github.com/huggingface/peft/issues/341#issuecomment-1519460307
                param.data = param.data.float()
        n_iterations = self.args.n_iterations
        n_warmup_iterations = int(0.1 * n_iterations)
        train_config = [
            {"params": [p for n, p in layer.named_parameters() if p.requires_grad],
                "lr": lr,
                "weight_decay": 0.,
            },
        ]
        sys.stdout.write(f"Layer {layer_idx+1}/{n_layers}: Reconstructing with {self.runner.config.peft_strategy}, {Utils.get_percentage_of_trainable_parameters(layer):.4f}% trainable params\n")
        if len(train_config[0]['params']) == 0:
            sys.stdout.write(f"Layer {layer_idx+1}/{n_layers}: No trainable parameters found. Skipping.\n")
            if len(Utils.get_layers_of_modules(layer)) != 0:
                peft_strategy.at_train_end(layer_subset=Utils.get_layers_of_modules(layer), keep_masks=keep_masks)
            return
        optimizer = SGD(train_config[0]['params'], lr=lr, momentum=float(self.args.momentum if self.args.momentum is not None else 0.9))\
            if self.args.optim == 'sgd' else AdamW(train_config)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup_iterations, num_training_steps=n_iterations)

        # loss function
        if self.args.loss_fn == "mse" or self.args.loss_fn is None:
            criterion = nn.MSELoss(reduction="mean").to(device)
        elif self.args.loss_fn == "cosine":
            criterion_ = nn.CosineSimilarity(dim=-1).to(device)
            criterion = lambda x, y: -1 * criterion_(x, y).mean()
        else:
            raise ValueError(f"Invalid loss function: {self.args.loss_fn}")
     
        gradScaler = torch.amp.GradScaler('cuda')
        layer.train()

        if self.args.log_train_loss:
            train_losses = []
            if val_args is not None:
                val_losses = []
        if self.args.log_grad_norm:
            grad_norms = []

        current_it = 0
        if self.args.gradient_accumulation_steps is None:
            self.args.gradient_accumulation_steps = 1
        accum_steps = self.args.gradient_accumulation_steps
        peft_strategy.iteration_getter = lambda: current_it

        for step in tqdm(range(1, n_iterations * self.args.gradient_accumulation_steps + 1, 1)):
            # reinitialize the train iterator if it reaches the end
            if step == 1 or (step - 1) % len_tensor_dataloader == 0:
                train_iterator = iter(zip(*zip_list))
            inputs, targets, amask, pids, pembeds, loss_m = next(train_iterator)
            inputs = inputs.to(device)

            # info for gradient accumulation
            last_batch = step == n_iterations
            need_update = last_batch or (step % accum_steps == 0)
            if step > n_iterations - (n_iterations % self.args.gradient_accumulation_steps):
                accum_steps = max(n_iterations % self.args.gradient_accumulation_steps, 1)

            with torch.amp.autocast('cuda'):
                kwargs = {}
                if pids is not None:
                    kwargs['position_ids'] = pids.to(inputs.device)
                if amask is not None:
                    kwargs['attention_mask'] = amask.to(inputs.device)
                if pembeds is not None:
                    kwargs['position_embeddings'] = (pembeds[0].to(inputs.device), pembeds[1].to(inputs.device))

                outputs = layer(inputs, **kwargs)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if outputs.shape != targets.shape:
                    outputs = outputs.reshape(targets.shape)
                targets = targets.to(outputs.device)

                # if we use padding tokens, we need to mask the loss
                if loss_m is not None and self.args.mask_pad_tokens:
                    loss_m = loss_m.to(outputs.device)
                    if outputs.shape[:len(loss_m.shape)] != loss_m.shape:
                        if loss_m.numel() == outputs.shape[0]:
                            loss_m = loss_m.flatten()
                        else:
                            outputs = outputs.reshape((*loss_m.shape, -1))
                            targets = targets.reshape((*loss_m.shape, -1))
                    loss = criterion(outputs[loss_m], targets[loss_m]) / accum_steps
                else:
                    loss = criterion(outputs, targets) / accum_steps

            gradScaler.scale(loss).backward()
            if need_update:
                gradScaler.unscale_(optimizer)
                gradScaler.step(optimizer)
                gradScaler.update()
                if self.args.log_grad_norm:
                    grad_norm = [p.grad.norm().item() for p in layer.parameters() if p.grad is not None]
                    grad_norms.append(sum(grad_norm))
                scheduler.step()
                optimizer.zero_grad()
                layer.zero_grad()

            current_it += 1
            if self.args.log_train_loss:
                train_losses.append(loss.cpu().item())
                if val_args is not None:
                    val_loss = Utils.evaluate_loss(*val_args, device=device, layer=layer, criterion=criterion, args=self.args)
                    val_losses.append(val_loss)
        torch.cuda.empty_cache()

        # reset the original data type
        for param in layer.parameters():
            if param.requires_grad:
                param.data = param.data.to(original_dtype)

        peft_strategy.at_train_end(layer_subset=Utils.get_layers_of_modules(layer), keep_masks=keep_masks)

        return None if not self.args.log_train_loss else train_losses, None if not self.args.log_train_loss else val_losses, None if not self.args.log_grad_norm else grad_norms

    def get_wrappers_and_hooks(self, layer_subset: dict) -> tuple[dict, list]:
        """Gets the wrappers and hooks needed for the pruning method."""
        wrappers, hooks = {}, []
        if self.prune_method in ["wanda", "sparsegpt", "ria", "wanda_sp", "flap"]:
            if self.prune_method == "flap":
                WrapperClass = FLAPWrapper
            else:
                WrapperClass = WandaWrapper if self.prune_method in ["wanda", "ria", "wanda_sp"] else SparseGPTWrapper
            for name in layer_subset:
                wrappers[name] = WrapperClass(layer_subset[name])
            def define_hook_fn(name):
                def hook_fn(_, inp, out):
                    wrappers[name].add_batch(inp[0].data, out.data)
                return hook_fn

            for name in wrappers:
                hooks.append(layer_subset[name].register_forward_hook(define_hook_fn(name)))

        return wrappers, hooks