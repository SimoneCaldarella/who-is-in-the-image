import os
import torch
from tutus.utils import Logger, __MODEL_ZOO__

# Privacy layers utils
from tutus.modeling.noise_injection import NoiseInjection
from tutus.modeling.token_pruning import TokenPruning
from transformers import AutoConfig

# Load model utils
from tutus.utils.dist import get_device_id




def load_perturbation_layer(model_name, cfg_method):
    if not getattr(cfg_method, "requires_perturbation", False):
        return None
    if cfg_method.name == "noise":
        return NoiseInjection(cfg_method.level, cfg_method.position)
    if cfg_method.name == "pruning":
        return TokenPruning(cfg_method.sparsity, cfg_method.position)
    raise ValueError(f"Unsupported paper perturbation: {cfg_method.name}")


def load_model(cfg_model, 
               cfg_method,
               pristine=False):
    """
    Load a model and its corresponding processor from the model zoo.
    Args:
        model_name (str): The name of the model to load.
    Returns:
        model: The loaded model.
        processor: The corresponding processor for the model.
    """
    
    if cfg_model.name not in __MODEL_ZOO__:
        raise ValueError(f"Model `{cfg_model.name}` not found in the model zoo.")
    
    model_id = cfg_model.get("hf_id") or __MODEL_ZOO__[cfg_model.name]

    # config = AutoConfig.from_pretrained(model_id, force_download=False)
    # dtype = config.torch_dtype if not full_precision else torch.float32
    dtype = torch.bfloat16


    if cfg_model.name == "llava_next":
        from transformers import LlavaNextProcessor
        from tutus.modeling.modelzoo import LlavaNextForConditionalGeneration, LlavaNextWithTokensPerturbation

        processor = LlavaNextProcessor.from_pretrained(model_id)

        if cfg_method.requires_perturbation and not pristine:
            perturbation_layer = load_perturbation_layer(model_name=cfg_model.name,
                                                         cfg_method=cfg_method)
            model = LlavaNextWithTokensPerturbation.from_pretrained(model_id, torch_dtype=dtype)
            model.attach_perturbation_layer(perturbation_layer)   
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)

    elif cfg_model.name == "llava_onevision" or cfg_model.name == "llava_onevision_smol":
        from transformers import AutoProcessor
        from tutus.modeling.modelzoo import LlavaOnevisionForConditionalGeneration, LlavaOnevisionWithTokensPerturbation

        processor = AutoProcessor.from_pretrained(model_id)
        
        if cfg_method.requires_perturbation and not pristine:
            perturbation_layer = load_perturbation_layer(model_name=cfg_model.name,
                                                         cfg_method=cfg_method)
            model = LlavaOnevisionWithTokensPerturbation.from_pretrained(model_id, torch_dtype=dtype)
            model.attach_perturbation_layer(perturbation_layer=perturbation_layer)
        else:
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)

    elif cfg_model.name == "qwen2vl" or cfg_model.name == "qwen2vl_smol" or cfg_model.name == "qwen2_5vl" or cfg_model.name == "qwen2_5vl_smol":
        from transformers import AutoProcessor
        from tutus.modeling.modelzoo import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, Qwen2VLWithTokensPerturbation, Qwen2_5VLWithTokensPerturbation

        min_pixels = 3136
        max_pixels = 512*28*28

        processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        if "qwen2_5vl" in cfg_model.name:
            if cfg_method.requires_perturbation and not pristine:
                perturbation_layer = load_perturbation_layer(model_name=cfg_model.name,
                                                             cfg_method=cfg_method)
                model = Qwen2_5VLWithTokensPerturbation.from_pretrained(model_id, torch_dtype=dtype)
                model.attach_perturbation_layer(perturbation_layer)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        else:
            if cfg_method.requires_perturbation and not pristine:
                perturbation_layer = load_perturbation_layer(model_name=cfg_model.name,
                                                             cfg_method=cfg_method)
                model = Qwen2VLWithTokensPerturbation.from_pretrained(model_id, torch_dtype=dtype)
                model.attach_perturbation_layer(perturbation_layer)
            else:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)

    else:
        raise ValueError(f"Model `{cfg_model.name}` not supported.")

    processor.tokenizer.padding_side = "left"

    return model, processor
