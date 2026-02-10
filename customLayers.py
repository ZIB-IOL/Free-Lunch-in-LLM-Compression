import abc
import random
import torch
from utilities import SelectiveMethods

from peft import LoraConfig, get_peft_model
import sys
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torch.linalg as linalg

from peft.tuners.lora import LoraLayer


class CustomDropout(nn.Module):
    def __init__(self, p=0.):
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(torch.full_like(x, 1 - self.p))  # Contains many 1s and few 0s (assuming self.p is small)
        return x * mask + (1 - mask)


class CustomLayerBaseClass(nn.Module):
    """Base class for custom layers, only for LoRA."""
    def __init__(self, **kwargs):
        super().__init__()
        self.lora_layer = kwargs['lora_layer']
        self.dropout_p = kwargs['dropout_p'] or 0.

    @abc.abstractmethod
    def _initialize(self) -> None:
        pass

    @abc.abstractmethod
    def forward(self, x) -> torch.Tensor:
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def set_effective_weights(self):
        """Sets the effective weights in-place to avoid memory issues."""
        pass


class PruneLoraLayer(CustomLayerBaseClass):
    """Regular LoRA during training, but prunes the B@A matrix before merging."""

    def _initialize(self):
        # Use standard LoRA initialization
        pass

    def forward(self, x):
        # Standard LoRA forward pass
        return self.lora_layer(x)

    @torch.no_grad()
    def set_effective_weights(self):
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight
        scaling = self.lora_layer.scaling['default']
        
        # Compute B@A directly into the shape of the original weight
        lora_contribution = scaling * (lora_B @ lora_A).view_as(self.lora_layer.weight)
        
        # Get the mask from the original weights
        mask = self.lora_layer.weight != 0
        lora_contribution.mul_(mask)
        
        # Add to the original weight in-place
        self.lora_layer.weight.add_(lora_contribution)


class PruneLoraLayerEfficient(PruneLoraLayer):
    """Regular LoRA during training, but prunes the B@A matrix before merging. Doesnt use dropout and hence has faster forward pass."""


    def forward(self, x):
        # Standard LoRA forward pass but without dropout
        original_weight = self.lora_layer.weight
        lora_A = self.lora_layer.lora_A['default']
        lora_B = self.lora_layer.lora_B['default']
        scaling = self.lora_layer.scaling['default']

        lora_contribution = (lora_B.weight @ lora_A.weight).reshape(original_weight.shape)   
        return F.linear(x, original_weight + scaling * lora_contribution, self.lora_layer.bias)

class CustomHadamardLoraLayer(CustomLayerBaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_dropout = CustomDropout(p=self.dropout_p)
        self._initialize()

    def _initialize(self):
        # Get LoRA matrices
        #scaling = self.lora_layer.scaling['default']
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight

        # Initialize B and A to 1./sqrt(r)
        with torch.no_grad():
            nn.init.ones_(lora_B)
            nn.init.ones_(lora_A)
            r = lora_A.shape[0]
            lora_A.data *= (1 / sqrt(r))
            lora_B.data *= (1 / sqrt(r))
            # We do not need to multiply by the scaling matrix, since we initialize B and A to 1./sqrt(r) and B@A = I (all ones matrix)

    def forward(self, x):
        original_weight = self.lora_layer.weight
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight

        # Compute LoRA contribution
        lora_contribution = (lora_B @ lora_A).reshape(original_weight.shape)

        # Apply custom dropout to lora_contribution
        lora_contribution = self.custom_dropout(lora_contribution)

        # Hadamard product
        effective_weight = original_weight * lora_contribution

        return F.linear(x, effective_weight, self.lora_layer.bias)
    
    @torch.no_grad()
    def set_effective_weights(self):
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight
        
        # Compute B@A directly into the shape of the original weight
        lora_contribution = (lora_B @ lora_A).view_as(self.lora_layer.weight)
        
        # Multiply the original weight in-place with the lora contribution
        self.lora_layer.weight.mul_(lora_contribution)

class CustomMaskedLoraLayer(CustomLayerBaseClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_masking_freq = kwargs['lora_masking_freq'] or 1
        self.masking_freq = self.initial_masking_freq
        self.total_iterations = kwargs.get('total_iterations', None)
        self.get_current_iteration_callback = kwargs['get_current_iteration_callback']
        self._initialize()

    def _initialize(self):
        # We use the same initialization as the original LoRA
        pass

    def forward(self, x):
        original_weight = self.lora_layer.weight
        lora_A = self.lora_layer.lora_A['default']
        lora_B = self.lora_layer.lora_B['default']
        scaling = self.lora_layer.scaling['default']
        
        lora_input = self.lora_layer.lora_dropout['default'](x)
        current_iteration = self.get_current_iteration_callback()() # This a function that returns the function returning the current iteration

        if current_iteration % self.masking_freq == 0:
            # Compute LoRA contribution
            lora_contribution = (lora_B.weight @ lora_A.weight).reshape(original_weight.shape)
            
            # Infer sparsity mask on the fly
            sparsity_mask = (original_weight != 0).to(dtype=lora_A.weight.dtype)   # Not using .float() since that will cast to float32

            # Apply sparsity mask to LoRA contribution
            masked_lora_contribution = lora_contribution * sparsity_mask
            del sparsity_mask

            lora_output = F.linear(lora_input, scaling * masked_lora_contribution, None)
        else:
            lora_output = scaling * lora_B(lora_A(lora_input))
        
        return F.linear(x, original_weight, self.lora_layer.bias) + lora_output

    @torch.no_grad()
    def set_effective_weights(self):
        lora_A = self.lora_layer.lora_A['default'].weight
        lora_B = self.lora_layer.lora_B['default'].weight
        scaling = self.lora_layer.scaling['default']
        
        # Compute B@A directly into the shape of the original weight
        lora_contribution = (lora_B @ lora_A).view_as(self.lora_layer.weight)
        
        # Apply sparsity mask and scaling in-place
        lora_contribution.mul_((self.lora_layer.weight != 0).to(dtype=lora_A.dtype)).mul_(scaling)
        
        # Add to the original weight in-place
        self.lora_layer.weight.add_(lora_contribution)

@torch.jit.script
def fused_lora_masked_matmul(B: torch.Tensor, A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # Compute matmul and reshape in one go
    result = (B @ A).view_as(W)
    # Zero out elements in-place where W is zero
    result.masked_fill_(W == 0, 0)
    return result

class MaskedLoraLayerEfficient(CustomMaskedLoraLayer):
    """Fused masking, single forward pass, no mask caching, no dropout possible."""

    def forward(self, x):
        original_weight = self.lora_layer.weight
        lora_A = self.lora_layer.lora_A['default']
        lora_B = self.lora_layer.lora_B['default']
        scaling = self.lora_layer.scaling['default']
        
        current_iteration = self.get_current_iteration_callback()() # This a function that returns the function returning the current iteration

        if current_iteration % self.masking_freq == 0:
            masked_lora_contribution = fused_lora_masked_matmul(lora_B.weight, lora_A.weight, original_weight)
            return F.linear(x, original_weight + scaling * masked_lora_contribution, self.lora_layer.bias)
        else:
            lora_output = scaling * lora_B(lora_A(x))
            return F.linear(x, original_weight, self.lora_layer.bias) + lora_output

@torch.jit.script
def fused_lora_spp_matmul(B: torch.Tensor, A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # Compute matmul and reshape in one go
    result = (B @ A).view_as(W)
    # Multiply by W in-place
    result.mul_(W)
    return result


class SPPLoraLayerEfficient(MaskedLoraLayerEfficient):
    """Instead of computing (W + M*B@A)x, we compute (W + W*B@A)x, where W is the original weight and M is the sparsity mask."""        

    def forward(self, x):
        original_weight = self.lora_layer.weight
        lora_A = self.lora_layer.lora_A['default']
        lora_B = self.lora_layer.lora_B['default']
        scaling = self.lora_layer.scaling['default']
        
        spp_lora_contribution = fused_lora_spp_matmul(lora_B.weight, lora_A.weight, original_weight)
        return F.linear(x, original_weight + scaling * spp_lora_contribution, self.lora_layer.bias)


