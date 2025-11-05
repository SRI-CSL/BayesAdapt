from peft.tuners.lora import LoraLayer, Linear, Linear4bit, Linear8bitLt

def wrap_lora_layers(module, wrapper_fn, target_modules=[]):
    for name, child in module.named_children():
        is_target = name in target_modules
        is_lora = isinstance(child, LoraLayer)
        is_linear = isinstance(child, Linear) or isinstance(child, Linear4bit) or isinstance(child, Linear8bitLt)
        if is_target and is_lora and is_linear:
            module._modules[name] = wrapper_fn(child)
        else:
            wrap_lora_layers(child, wrapper_fn, target_modules)
