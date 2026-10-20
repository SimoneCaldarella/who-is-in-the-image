import torch

from tutus.utils.dist import unique_generator

from tutus.dataset import TutusPD, TutusTS, MLLMUTS
from tutus.utils.dist import use_distributed, get_rank

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def load_train_dataset(model, config, seed=42):
    if config.name == "TUTUS-PD":
        #try:
        train_dataset = TutusPD(config.root, 
                                model,
                                config.num_samples_per_class, 
                                config.num_unique_classes, 
                                config.unique_images,
                                seed=seed)
        return train_dataset
    

        # except:
            # return False
    else:
        raise ValueError(f"Unknown dataset: {config.name}")
    


def load_test_dataset(config, pristine_path=None, seed=42):
    if "TUTUS-TS" in config.name:
        # This include both TUTUS-TS and TUTUS-TS-P, where P stands
        # for the version with paraphrases of "Who is the celebrity in the picture?"
        test_dataset = TutusTS(config.name, 
                               config.root,  
                               config.num_samples_per_class,  
                               config.num_unique_classes, 
                               filter_with_pristine=pristine_path, 
                               seed=seed)
        return test_dataset
    
    elif "TUTUS-DEV" in config.name: 
        test_dataset = TutusTS(config.name,
                               config.root,
                               config.num_samples_per_class,
                               config.num_unique_classes,
                               filter_with_pristine=pristine_path,
                               seed=seed)
        return test_dataset

    elif "MLLMU" in config.name: 
        test_dataset = MLLMUTS(config.name,
                               config.root,
                               seed=seed)
        return test_dataset

    else:
        raise ValueError(f"Unknown dataset: {config.name}")
    

def load_test_dataloader(test_dataset, batch_size):
    if use_distributed():
        test_sampler = DistributedSampler(test_dataset, shuffle=False, rank=get_rank())

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, 
                                 sampler=test_sampler if use_distributed() else None, shuffle=False)
    
    return test_dataloader

def load_train_valid_dataloader(train_dataset, train_batch_size, test_batch_size, seed=42):
    valid_size = int(min(100, int(0.05 * len(train_dataset))))
    train_size = int(len(train_dataset) - valid_size)
    
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size], generator=unique_generator(seed)
    )

    if use_distributed():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, rank=get_rank())
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False, rank=get_rank())

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=1, 
                                    sampler=train_sampler if use_distributed() else None)
    valid_dataloader = DataLoader(valid_dataset, batch_size=test_batch_size, num_workers=1,
                                    sampler=valid_sampler if use_distributed() else None)
    
    return train_dataloader, valid_dataloader