import abc
import torch
from customLayers import CustomHadamardLoraLayer, CustomMaskedLoraLayer, MaskedLoraLayerEfficient, PruneLoraLayerEfficient, SPPLoraLayerEfficient, PruneLoraLayer
from utilities import DictAccessor, SelectiveMethods

from peft import LoraConfig, get_peft_model
import sys
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torch.linalg as linalg
import torch.nn.utils.prune as prune

from peft.tuners.lora import LoraLayer

class PeftMethodBaseClass:
    """PEFT method base class - Important: Do not activate layers that have been pruned without keeping the mask, otherwise sparsity is destroyed."""
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.runner = kwargs['runner']
        self.config = kwargs['config']
        self.is_reconstruct = kwargs['is_reconstruct']

        self.iteration_getter = None    # Must be set, otherwise asserts

    @abc.abstractmethod
    def select_peft_layers(self, **kwargs):
        pass

    def at_train_end(self):
        """This is called at the end of training, to clean up the model and remove the peft layers, if possible."""
        pass

    def get_current_iteration_callback(self):
        assert self.iteration_getter is not None, "Iteration getter must be set."
        return self.iteration_getter

class FullFT(PeftMethodBaseClass):
    def at_train_end(self, layer_subset: dict | None = None, keep_masks: bool = False):
        if not keep_masks:
            # Remove the masks
            if layer_subset is None:
                for name, module in self.model.named_modules():
                    if prune.is_pruned(module) and hasattr(module, 'weight'):
                        prune.remove(module, name='weight')
            else:
                for name in layer_subset:
                    if prune.is_pruned(layer_subset[name]):
                        prune.remove(layer_subset[name], name='weight')

    def select_peft_layers(self, **kwargs):
        SelectiveMethods.activate_model(self.model)

class BlockOnlyFullFT(PeftMethodBaseClass):
    def at_train_end(self, layer_subset: dict | None = None, keep_masks: bool = False):
        if not keep_masks:
            # Remove the masks
            if layer_subset is None:
                for name, module in self.model.named_modules():
                    if prune.is_pruned(module) and hasattr(module, 'weight'):
                        prune.remove(module, name='weight')
            else:
                for name in layer_subset:
                    if prune.is_pruned(layer_subset[name]):
                        prune.remove(layer_subset[name], name='weight')

    def select_peft_layers(self, **kwargs):
        # First deactivate everything
        SelectiveMethods.deactivate_model(self.model)
        
        # Then selectively activate only transformer blocks
        base_model = self.model
        while hasattr(base_model, "model"):
            base_model = base_model.model
            
        # Identify transformer blocks based on model architecture
        if hasattr(base_model, 'decoder'):
            # OPT case
            for layer in base_model.decoder.layers:
                SelectiveMethods.activate_specific_modules([layer])
        elif hasattr(base_model, 'layers'):
            # Llama/Mistral case
            for layer in base_model.layers:
                SelectiveMethods.activate_specific_modules([layer])
        else:
            raise ValueError("Could not identify transformer blocks structure.")

class SelectivePEFT(PeftMethodBaseClass):
    """Base class for LoRA, to be used for all methods with slightly differing reparametrizations. Uses LN parameters + biases + LoRA on pruned layers, as well as lm head."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # PEFT Configuration
        self.use_peft_dict = {
            'biases': self.config.peft_use_bias or False,
            'layer_norm_params': self.config.peft_use_ln or False,
            'lm_head': self.config.peft_use_lm_head or False,
            'lora': self.config.peft_use_lora or False,
        }
        assert sum(self.use_peft_dict.values()) > 0, "At least one of the peft_use_* must be True."

        # LoRA Configuration
        self.lora_r = self.config.lora_r
        self.lora_alpha = self.config.lora_alpha
        self.lora_dropout = self.config.lora_dropout
        self.lora_masking_freq = self.config.lora_masking_freq
        self.lora_type = self.config.lora_type

        self.CustomModule = None # To be set by below
        self.non_pruned_lora_layers = []    # Layers that are not pruned and hence use classical LoRA
        self.total_iterations = kwargs['total_iterations']        

        self.target_modules = []
        if self.use_peft_dict['lm_head']:
            if not self.is_reconstruct:
                sys.stdout.write("SelectivePEFT: Using LM head.\n")
                self.target_modules.append("lm_head")
                self.non_pruned_lora_layers.append("lm_head")
            else:
                sys.stdout.write("lm_head has been specified for reconstruction, but it is not pruned. Skipping.\n")
        if self.use_peft_dict['lora']:
            sys.stdout.write("SelectivePEFT: Using LoRA.\n")
            # LoRA Configuration    
            self.target_modules = self.target_modules + [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "out_proj",     # OPT
                    "o_proj",       # Mistral/Llama
                    "gate_proj",    # Mistral/Llama
                    "up_proj",      # Mistral/Llama
                    "down_proj",    # Mistral/Llama
                    "fc1",          # OPT
                    "fc2",          # OPT
                    ]
            assert self.lora_type in ['lora', 'lora_prune', 'hadamard', 'masked', 'masked_efficient', 'spp', 'lora_prune_efficient'], "LoRA type must be in ['lora', 'lora_prune', 'hadamard', 'masked', 'masked_efficient', 'spp', 'lora_prune_efficient']."
            if self.lora_type == 'hadamard':
                self.CustomModule = CustomHadamardLoraLayer
                if self.lora_dropout > 0:
                    sys.stdout.write(f"Warning:CustomDropout: Dropout probability specified as {self.lora_dropout}, setting this to zero since it is not clear whether this works with the Hadamard product.\n")
                    self.lora_dropout = 0.
            elif self.lora_type == 'masked':
                self.CustomModule = CustomMaskedLoraLayer
            elif self.lora_type == 'masked_efficient':
                if self.lora_dropout > 0:
                    sys.stdout.write(f"Warning:CustomDropout: Dropout probability specified as {self.lora_dropout}, setting this to zero since this does not work with the efficient implementation.\n")
                    self.lora_dropout = 0.
                self.CustomModule = MaskedLoraLayerEfficient
            elif self.lora_type == 'spp':
                if self.lora_dropout > 0:
                    sys.stdout.write(f"Warning:CustomDropout: Dropout probability specified as {self.lora_dropout}, setting this to zero since this does not work with the efficient implementation.\n")
                    self.lora_dropout = 0.
                if self.lora_masking_freq != 1:
                    sys.stdout.write(f"Warning:SPP LoRA is only supported with masking frequency 1, setting this to 1.\n")
                    self.lora_masking_freq = 1
                self.CustomModule = SPPLoraLayerEfficient
            elif self.lora_type == 'lora_prune':
                self.CustomModule = PruneLoraLayer
            elif self.lora_type == 'lora_prune_efficient':
                if self.lora_dropout > 0:
                    sys.stdout.write(f"Warning:CustomDropout: Dropout probability specified as {self.lora_dropout}, setting this to zero since this does not work with the efficient implementation.\n")
                    self.lora_dropout = 0.
                self.CustomModule = PruneLoraLayerEfficient
    
    def get_lora_config(self) -> LoraConfig:
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        return config
    
    def select_peft_layers(self, **kwargs):
        SelectiveMethods.deactivate_model(self.model)

        if len(self.target_modules) > 0:
            config = self.get_lora_config()
            get_peft_model(self.model, config)

            # Replace LoRA layers with our custom implementation
            self._replace_lora_layers(self.model)

            if 'layer' in kwargs and kwargs['layer'] is not None:
                # Remove lora for all layers except the one specified
                self._remove_lora_layers_except_layer(self.model, kwargs['layer'])

        if self.use_peft_dict['biases']:
            sys.stdout.write("SelectivePEFT: Using biases.\n")
            SelectiveMethods.activate_biases(self.model)
        if self.use_peft_dict['layer_norm_params']:
            sys.stdout.write("SelectivePEFT: Using LN.\n")
            SelectiveMethods.activate_layer_norm_params(self.model)

    def _replace_lora_layers(self, model):
        """Replace the LoRA layers with the custom implementation."""
        for name, module in model.named_children():
            if isinstance(module, LoraLayer):
                if name in self.non_pruned_lora_layers:
                    # We skip the layers that use classical LoRA and are not pruned
                    continue
                elif self.CustomModule is not None:
                    setattr(model, name, self.CustomModule(
                        lora_layer=module, 
                        dropout_p=self.lora_dropout,
                        lora_masking_freq=self.lora_masking_freq,
                        total_iterations=self.total_iterations,
                        get_current_iteration_callback=self.get_current_iteration_callback
                    ))
            else:
                self._replace_lora_layers(module)

    def _remove_lora_layers_except_layer(self, model: torch.nn.Module, layer: torch.nn.Module):
        """Remove the LoRA layers except the one specified."""
        for name, module in model.named_children():
            if id(module) == id(layer):
                continue
            if self.CustomModule is not None and isinstance(module, self.CustomModule):
                setattr(model, name, module.lora_layer.base_layer)
            elif name in self.non_pruned_lora_layers:
                setattr(model, name, module.base_layer)
            else:
                self._remove_lora_layers_except_layer(module, layer)

    def at_train_end(self):
        self._merge_lora_layers(self.model)
    
    def _merge_lora_layers(self, model):
        """Merge the LoRA layers into the base model."""
        for name, module in model.named_children():
            if self.CustomModule is not None and isinstance(module, self.CustomModule):
                module.set_effective_weights()

                # Remove the custom LoRA layer and set the original one back
                setattr(model, name, module.lora_layer.base_layer)
                del module
            elif name in self.non_pruned_lora_layers:
                # We used classical LoRA for the embedding layer and classification layer, since not pruned, and we merge it in the classical way
                module.merge()
                setattr(model, name, module.base_layer)
                del module
            else:
                self._merge_lora_layers(module)