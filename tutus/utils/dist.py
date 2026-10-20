import torch.distributed as tdist
import os
import torch
import json
import shutil
import pandas as pd
from rich.progress import track
from tutus.utils.misc import aggregate_csvs, aggregate_jsons, aggregate_files

def barrier():
    # Synchronize all processes
    if is_distributed_available_and_initialized():
        tdist.barrier()
    else:
        pass


def use_distributed():
    # Check if running in distributed mode (torchrun used or not)
    return bool(os.environ.get("RANK", 0))


def is_distributed_available_and_initialized():
    return tdist.is_available() and tdist.is_initialized()


def get_rank():
    if is_distributed_available_and_initialized():
        return tdist.get_rank()
    return 0  # default to rank 0 if not distributed

def get_device_id():
    device_id = get_rank() % torch.cuda.device_count()
    return device_id


def get_world_size():
    if is_distributed_available_and_initialized():
        return tdist.get_world_size()
    return 1  # default to 1


def is_main_process():
    return get_rank() == 0


def create_global_id():
    # Create a unique ID for each run
    # This can be a random number or a UUID
    import uuid
    if use_distributed():
        if tdist.get_rank() == 0:
            uuid = uuid.uuid4()
            objects = [uuid] # any picklable object
        else:
            objects = [None]
        # Assumes backend is not NCCL
        device = torch.device(get_device_id())
        tdist.broadcast_object_list(objects, src=0, device=device)
        return objects[0]
    else:
        return uuid.uuid4()
    
def gather_dataframes_across_ranks(local_df):
    """Gathers pandas DataFrames from all ranks and merges them on rank 0."""
    world_size = tdist.get_world_size()
    gathered = [None for _ in range(world_size)]
    tdist.all_gather_object(gathered, local_df)
    return gathered
    
    
def unique_generator(seed=42):
    generator = torch.Generator().manual_seed(seed)
    return generator

def setup_distributed(return_generator=True):
    # Setup the distributed environment
    torch.manual_seed(42)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    tdist.init_process_group("nccl")
    
    if return_generator: return_generator
    else: return None


def cleanup_distributed():
    if is_distributed_available_and_initialized():
        tdist.destroy_process_group()


def save_files_from_ranks(data, experiment_folder, filename, format="csv"):
    if use_distributed():
        # Given a root_folder and a filename, save the file from each rank
        # like folder/filename_RANK[ID].json
        # Each run has its own temp ID generated when it start the program
        temp_folder = os.path.join(experiment_folder, "temp")
        os.makedirs(temp_folder, exist_ok=True)
        assert filename.count(".") == 1, "Filename should contain only one dot, found {} in {}".format(filename.count("."), filename)
        rank_filename = filename.split(".")[0] + "_RANK" + str(get_rank()) + "." + filename.split(".")[1]
        rank_filename = os.path.join(temp_folder, rank_filename)
    else:
        rank_filename = os.path.join(experiment_folder, filename)

    if "csv" in format:
        assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"
        data.to_csv(rank_filename, index=False)

    elif "json" in format:
        assert isinstance(data, list), "Data should be a list"
        with open(rank_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    else:
        assert isinstance(data, list), "Data should be a string"
        with open(rank_filename, 'w', encoding='utf-8') as f:
            for line in data:
                f.write(line + "\n")

    return rank_filename


def aggregate_files_from_ranks(experiment_folder, filename):
    # Aggregate files from all ranks into a single file
    # All files should have the same name on each rank
    # like folder/temp/filename_RANK[ID].json
    # and I want
    tdist.barrier()
    temp_folder = os.path.join(experiment_folder, "temp")

    if is_main_process():
        # Get all files from all ranks
        files = []
        for rank in range(get_world_size()):
            rank_filename = filename.split(".")[0] + "_RANK" + str(rank) + "." + filename.split(".")[1]
            files.append(os.path.join(temp_folder, rank_filename))

        # aggregate files
        if files[0].endswith(".json"):
            aggregate_jsons(files, os.path.join(experiment_folder, filename))

        elif files[0].endswith(".csv"):
            aggregate_csvs(files, os.path.join(experiment_folder, filename))

        else:
            aggregate_files(files, os.path.join(experiment_folder, filename))

        # Remove temp folder
        shutil.rmtree(temp_folder, ignore_errors=True)  


def progress(data, *args, **kwargs):  
    disable = False

    if "disable" in kwargs:
        disable = kwargs["disable"]
        del kwargs["disable"]

    
    if is_main_process():
        
        if disable:
            return data
        else:
            return track(data, *args, **kwargs)
    else:
        return data
    
        
        


