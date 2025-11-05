import copy
import torch
from typing import Any
from peft.tuners.lora import LoraLayer
from .lorawrapper import LoraWrapper

#In order to have deep ensemble fit into the larger sampling based framework, use a single wrapper that cycles through the ensemble members
#This means that during training and inference, the model MUST be called ensemble_size times
#During training we backprop the mean of each ensemble member's loss rather than train each member independently
#Unclear what effect this has on learning dynamics, but might actually be beneficial?
class DeepEnsembleLoraWrapper(LoraWrapper):
    def __init__(self, lora_layer: LoraLayer, ensemble_size: int = 3, *args: Any, **kwargs: Any):
        super().__init__(lora_layer, *args, **kwargs)
        assert 'default' in self.lora_A.keys(), "Only support default adapter for DeepEnsembleLoraWrapper."
        self.adapter_names = ['default']
        for i in range(1, ensemble_size):
            new_key = f'default_{i}'
            self.adapter_names.append(new_key)
            self.lora_A[new_key] = torch.nn.Linear( #init new loraA otherwise the members won't be distinct
                in_features=self.lora_A['default'].in_features,
                out_features=self.lora_A['default'].out_features,
                bias=self.lora_A['default'].bias is not None,
                dtype=self.lora_A['default'].weight.dtype,
            )
            self.lora_B[new_key] = copy.deepcopy(self.lora_B['default'])
            self.lora_dropout[new_key] = copy.deepcopy(self.lora_dropout['default'])
            self.scaling[new_key] = self.scaling['default'] #just a scalar so no need to deepcopy
            self.lora_dropout[new_key] = copy.deepcopy(self.lora_dropout['default'])
        self.active_adapters = ['default']
        self.current_index = 0

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        output = super().forward(x, *args, **kwargs)
        self.current_index = (self.current_index + 1) % len(self.adapter_names)
        self.active_adapters = [self.adapter_names[self.current_index]]
        return output
