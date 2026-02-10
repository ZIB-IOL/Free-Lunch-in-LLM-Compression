import sys
from typing import NamedTuple 
import torch
from utilities import Utils, PruneUtils, FLAPWrapper, compress


class PruneFLAP:

    def __init__(self, runner, args: NamedTuple, prune_method: str, model: torch.nn.Module, prune_n: int = 0, prune_m: int = 0):
        self.runner = runner
        self.args = args
        self.prune_method = prune_method
        self.model = model
        self.prune_n = prune_n
        self.prune_m = prune_m       

        assert self.prune_method == "flap", "Invalid pruning method."
        self.requires_calibration_wrapper = True

        self.reconstruction_error_initial, self.reconstruction_error_final = None, None

    @torch.no_grad()
    def prune(self, do_prune: bool = True):
        device = self.args.device
        pruning_block_size = self.args.pruning_block_size or self.args.block_size
        layers_prune = Utils.get_reconstruction_layers(model=self.model, block_size=pruning_block_size, rec=False, runner=self.runner)

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

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
                                        
        
        with torch.no_grad():
            inps_prune, outs_prune, attention_mask, position_ids, position_embeddings, loss_mask =\
                PruneUtils.prepare_calibration_input(self.args, self.model, dataloader, device, pad_token_id=self.runner.tokenizer.pad_token_id,
                                                        mask_pad_tokens=self.args.mask_pad_tokens)

        wrappers, hooks = {}, []
        attn_metric_list = []
        attn_baseline_inp_list = []
        mlp_metric_list = []
        mlp_baseline_inp_list = []
        for i_prune, layer_prune in enumerate(layers_prune):
            sys.stdout.write(f"FLAP processing layer {i_prune+1} of {len(layers_prune)}.\n")
            subset = Utils.get_layers_of_modules(layer_prune)
            # remove all linear layers except o_proj and down_proj/fc2
            keys_to_remove = []
            for key in subset.keys():
                if not "o_proj" in key and not "out_proj" in key and not "down_proj" in key and not "fc2" in key:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                subset.pop(key)

            if self.requires_calibration_wrapper:
                if hasattr(self.model, "hf_device_map") and f"self.model.layers.{i_prune}" in self.model.hf_device_map:
                    # handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                    device = self.model.hf_device_map[f"self.model.layers.{i_prune}"]

                wrappers, hooks = self.get_wrappers_and_hooks(layer_subset=subset)
                # Changes the outs dynamically
                Utils.get_outputs(args=self.args, layer=layer_prune, inps=inps_prune, outs=outs_prune,
                                    attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings,
                                    loss_mask=loss_mask, device=device)
                for h in hooks:
                    h.remove()
            
            metrics = {
                'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
                'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
                'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
            }

            for name in subset:
                W = subset[name].weight.data

                if "o_proj" in name or "out_proj" in name:
                    W_metric = metrics[self.runner.config.flap_metric or "WIFV"](wrappers, subset, name) ** 2
                    attn_metric_list.append(W_metric.cpu())
                    attn_baseline_inp_list.append(wrappers[name].baseline_inp.type(torch.half))
                else:
                    W_metric = metrics[self.runner.config.flap_metric or "WIFV"](wrappers, subset, name)
                    mlp_metric_list.append(W_metric.cpu())
                    mlp_baseline_inp_list.append(wrappers[name].baseline_inp.type(torch.half))
                wrappers[name].free()

            inps_prune, outs_prune = outs_prune, inps_prune
            torch.cuda.empty_cache()

        standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

        attn_metric = torch.stack(attn_metric_list)
        attn_metric = standarlization(attn_metric)
        attn_metric = attn_metric.reshape(len(layers_prune), -1, 128).mean(dim=2)
        
        mlp_metric = torch.stack(mlp_metric_list)
        mlp_metric = standarlization(mlp_metric)
            
        prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
        sorted_prune, indices = torch.sort(prune_metric, descending=True)
        compression_weight = torch.ones_like(indices)
        compression_weight[indices < attn_metric.numel()] = 512.0 / 3
        threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - self.runner.config.flap_pruning_ratio)))]
        attn_mask = (attn_metric > threshold)
        mlp_mask = (mlp_metric > threshold)
        
        for idx in range(len(layers_prune)):
            if f"model.layers.{idx}" in getattr(self.model, 'hf_device_map', {}): 
                compress(layers_prune[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None,
                         self.model.hf_device_map[f"model.layers.{idx}"], bias=self.model.model.layers[idx].self_attn.o_proj.bias is not None,
                         unstr=self.runner.config.flap_unstr or False)
            else:
                compress(layers_prune[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, device,
                         bias=self.model.model.layers[idx].self_attn.o_proj.bias is not None,
                         unstr=self.runner.config.flap_unstr or False)
                    
            if f"model.layers.{idx}" in getattr(self.model, 'hf_device_map', {}): 
                compress(layers_prune[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx],
                         self.model.hf_device_map[f"model.layers.{idx}"], bias=self.model.model.layers[idx].mlp.down_proj.bias is not None,
                         unstr=self.runner.config.flap_unstr or False)
            else:
                compress(layers_prune[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], device,
                         bias=self.model.model.layers[idx].mlp.down_proj.bias is not None,
                         unstr=self.runner.config.flap_unstr or False)
                    
        self.model.config.use_cache = use_cache 
        torch.cuda.empty_cache()

        self.train_losses = None
        self.val_losses = None
        self.grad_norms = None

    def get_wrappers_and_hooks(self, layer_subset: dict) -> tuple[dict, list]:
        """Gets the wrappers and hooks needed for the pruning method."""
        wrappers, hooks = {}, []
        for name in layer_subset:
            wrappers[name] = FLAPWrapper(layer_subset[name], self.runner.config.flap_metric)
        def define_hook_fn(name):
            def hook_fn(_, inp, out):
                wrappers[name].add_batch(inp[0].data, out.data)
            return hook_fn

        for name in wrappers:
            hooks.append(layer_subset[name].register_forward_hook(define_hook_fn(name)))

        return wrappers, hooks