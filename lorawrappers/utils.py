from peft.tuners.lora import LoraLayer, Linear

def wrap_lora_layers(module, wrapper_fn, target_modules=[]):
    for name, child in module.named_children():
        if name in target_modules and isinstance(child, LoraLayer) and isinstance(child, Linear):
            module._modules[name] = wrapper_fn(child)
        else:
            wrap_lora_layers(child, wrapper_fn, target_modules)
