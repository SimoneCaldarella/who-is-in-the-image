import random
import json
import os
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset

class MLLMUTS(Dataset):

    def __init__(self,
                 dataset_name="MLLMU-TS-P",
                 dataset_root="./mllmu/",
                 seed=42):
        
        # Here I am not using the same pristine path as it is not needed due to
        # the few number of samples of the dataset (~500)
        self.name = dataset_name
        self.root = dataset_root
        self.images_root = os.path.join(dataset_root, "images") # Root directory of images
        self.annotations = os.path.join(dataset_root, "annotations", f"{self.name}.csv") # Root directory of annotations
        self.seed = seed
        self.samples = self._load_dataset()

    def _load_dataset(self):
        # Load the dataset
        df = pd.read_csv(self.annotations)
        df["image_path"] = df["image_path"].apply(lambda x: f"{self.images_root}/{x}")
        df["leaking_prompt"] = ["What is the name of this person?" for _ in range(len(df))]

        return df.values.tolist()

    def _num_unique_images(self):
        return len(set([sample[1] for sample in self.samples]))
    
    def _num_unique_classes(self):
        return len(set([sample[0] for sample in self.samples]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __str__(self):
        idx = random.randint(0, self.__len__()-1)
        sample_str = '",\n\t"'.join(self.samples[idx]) + "\""

        return f"Tutus TEST \
                \n * Number of samples: {self.__len__()} \
                \n * Number of unique images: {self._num_unique_images()} \
                \n * Number of unique classes: {self._num_unique_classes()} \
                \n * Sample example n. {idx}: [\n\t\"{sample_str}\n]"