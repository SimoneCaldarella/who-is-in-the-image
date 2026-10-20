import os
import random
import numpy as np
import torch
import pandas as pd
import json
import argparse
from omegaconf import OmegaConf, DictConfig
from tutus.utils.const import __TRAINING_FREE__



def hist(data, bins=10, width=50):
    if not data:
        print("No data provided.")
        return

    min_val, max_val = min(data), max(data)
    bin_size = (max_val - min_val) / bins
    bin_counts = [0] * bins

    # Count how many items fall into each bin
    for value in data:
        index = int((value - min_val) / bin_size)
        if index == bins:  # Edge case for the max value
            index -= 1
        bin_counts[index] += 1

    max_count = max(bin_counts)

    for i, count in enumerate(bin_counts):
        bar = '#' * int(count / max_count * width)  # Scale bar length
        bin_range_start = min_val + i * bin_size
        bin_range_end = bin_range_start + bin_size
        print(f"{bin_range_start:6.2f} - {bin_range_end:6.2f} | {bar} ({count})")


def setup_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def aggregate_jsons(filenames, output_file):
    merged = []

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert isinstance(data, list), f"Skipping {filename}: not a JSON array"
            merged.extend(data)

    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(merged, out)


def aggregate_csvs(filenames, output_file):
    # Aggregate content from list of files into a single object
    # and save it to a single CSV file
    aggregated_df = pd.DataFrame()
    for filename in filenames:
        temp_df = pd.read_csv(filename)
        aggregated_df = pd.concat([aggregated_df, temp_df], ignore_index=True)
    
    aggregated_df.to_csv(output_file, index=False)


def aggregate_files(filenames, output_file):
    # Aggregate content from list of files into a single object
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
    

def conversation(system_prompt=None, question=None, answer=None):
    conversation_template_filled = []
    if system_prompt is not None:
        conversation_template_filled.append(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": f"{system_prompt} "},
                ],
            }
        )
        
    if question is not None:
        conversation_template_filled.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question} "},
                ],
            }
        )
    if answer is not None:
        conversation_template_filled.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{answer} "},
                ],
            }
        )
    return conversation_template_filled


def get_model_memory_usage(model, unit="b"):
    total_memory = 0
    for param in model.parameters():
        tensor_memory = get_tensor_memory_usage(param, unit)
        total_memory += tensor_memory
    return total_memory

def get_tensor_memory_usage(tensor, unit="b"):
    """
    Get the memory usage of a tensor in bytes.
    """
    units = {
        "b": 1,
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
    }
    if unit not in units:
        raise ValueError(f"Invalid unit: {unit}. Choose from {list(units.keys())}.")
    return tensor.element_size() * tensor.nelement() / units[unit]


def freeze_model(model, excludes=["privacy_layer"]):
        for name, param in model.named_parameters():
            param.requires_grad = False

        for name, params in model.named_parameters():
            if "next" in model.config.model_type and "vision_model.encoder.layers.23" in name:
                continue
            for exc in excludes:
                if exc in name:
                    params.requires_grad = True
        return model

def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total > 0 else 0
    return f"Total: {total:.2e}, Trainable: {trainable:.2e}, Ratio: {ratio:.2%}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file, inside ./configs folder")
    args, unknown = parser.parse_known_args()
    # Both -- and nothing before the argument are valid, so we remove the leading dashes
    unknown = [arg.lstrip("-") for arg in unknown]
    return args, unknown

def load_config() -> DictConfig:
    args, overrides = parse_args()
    base_cfg = OmegaConf.load(args.config)
    override_cfg = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.merge(base_cfg, override_cfg)
    
    return cfg


class Color:
    @staticmethod
    def magenta(text):
        return f"[bold magenta]{text}[/bold magenta]"

            

