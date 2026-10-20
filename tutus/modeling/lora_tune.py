
import torch
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

def setup_lora(model, lora_name, include=None):

    final_target_modules = []

    if "next" in str(model.config.model_type.lower()):
        special_case_exclude = "vision_model.encoder.layers.23"
    else:
        special_case_exclude = None
    
    # Detect submodule in main model
    for name, nm in model.named_modules():
        if special_case_exclude and special_case_exclude in name:
            continue
        added = False
        for module in include:
            if not added and module in name:
                if isinstance(nm, torch.nn.Linear):
                    final_target_modules.append(name)

                added = True

    if 'lm_head' in final_target_modules: # needed for 16-bit
        final_target_modules.remove('lm_head')



    lora_config = LoraConfig(
        init_lora_weights="gaussian",
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=final_target_modules,
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config, lora_name)
    model.set_adapter(lora_name)

    return model

def load_lora(model, lora_path, name):
    model = PeftModel.from_pretrained(
        model,
        lora_path+"/"+name,
        adapter_name=name,
        strict=False,
        is_trainable=False,
    )
    model.set_adapter(name)

    return model

def save_lora(model, lora_path):
    model.save_pretrained(lora_path)
