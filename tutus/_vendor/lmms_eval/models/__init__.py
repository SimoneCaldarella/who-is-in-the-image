from importlib import import_module

_MODELS = {"llava_hf": "LlavaHf", "qwen2_vl": "Qwen2_VL", "qwen2_5_vl": "Qwen2_5_VL"}

def get_model(name):
    if name not in _MODELS:
        raise ValueError(f"Supported utility adapters: {', '.join(_MODELS)}")
    return getattr(import_module(f"lmms_eval.models.{name}"), _MODELS[name])
