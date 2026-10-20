import random
import json
import os
import pandas as pd
from torch.utils.data import Dataset
from tutus.utils.const import __BROKEN_IMAGES__, __LEAKING_PROMPTS__


class TutusPD(Dataset):
    """Dataset class for Tutus Preference Dataset.
    Each sample is a quadruplet of (image_path, question, chosen, rejected)
    depending on whether you are using V0 (template based chosen) or
    V1 (retrieved image chosen). V2 will include blurred anonymization.
    """
    def __init__(self, 
                 dataset_root="./celebdata", 
                 model="qwen", 
                 num_samples_per_class=-1, 
                 num_unique_classes=-1, 
                 unique_images=True, 
                 seed=42):
        
        self.root = dataset_root
        self.model = model
        self.images_root = os.path.join(dataset_root, "train") # Root directory of images
        self.annotations = os.path.join(dataset_root, "annotations", f"TUTUS-PD-V1-{model}.csv") # Root directory of annotations

        if not os.path.exists(self.annotations):
            raise FileNotFoundError(f"Annotations file {self.annotations} not found. Please generate it first.")

        self.num_samples_per_class = num_samples_per_class
        self.num_unique_classes = num_unique_classes
        self.unique_images = unique_images
        self.seed = seed
        self.debug = bool(os.getenv("DEBUG", False))
        self.samples = self._load_dataset(num_samples_per_class, num_unique_classes, unique_images)
        self.classes = list(set([sample[1] for sample in self.samples]))
        
    def _load_dataset(self, num_samples_per_class=-1, num_unique_classes=-1, unique_images=False):
        # Load the dataset
        columns = ["idx", "name", "image_path", "question", "chosen", "rejected", "support"]
        df = pd.read_csv(self.annotations)[columns]
        df = df[~df["image_path"].isin(__BROKEN_IMAGES__)]
        df["image_path"] = df["image_path"].apply(lambda x: f"{self.images_root}/{x}")

        if num_unique_classes != -1:
            K = min(num_unique_classes, len(df.name.unique()))
            df = df[df["name"].isin(random.sample(df.name.unique().tolist(), k=K))]
        
        if num_samples_per_class != -1:
            df = df.groupby("name", group_keys=False).apply(lambda x: x.sample(n=min(num_samples_per_class, len(x)), 
                                                                               random_state=self.seed))

        if unique_images:
            df = df.drop_duplicates(subset=["image_path"], keep="first")
        
        df["image_path"] = df["image_path"].apply(lambda x: x.replace("//", "/"))

        if self.debug:
            return random.sample(df.values.tolist(), k=min(256, len(df)))
        return df.values.tolist()
    
    def _num_unique_images(self):
        return len(set([sample[2] for sample in self.samples]))
    
    def _num_unique_classes(self):
        return len(set([sample[1] for sample in self.samples]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __str__(self):
        idx = random.randint(0, self.__len__()-1)
        sample_str = '",\n\t"'.join(self.samples[idx][1:]) + "\""

        return f"Tutus Preference Dataset for ({self.model}) \
                \n * Number of samples: {self.__len__()} \
                \n * Number of unique images: {self._num_unique_images()} \
                \n * Number of unique classes: {self._num_unique_classes()} \
                \n * Sample example n. {idx}: [\n\t\"{sample_str}\n]"
    



class TutusTS(Dataset):
    """TUTUS Test set benchmark.
    """
    def __init__(self, 
                 dataset_name="TUTUS-TS-P",
                 dataset_root="./celebdata", 
                 num_samples_per_class=-1, 
                 num_unique_classes=-1,
                 filter_with_pristine=None, 
                 seed=42
                 ):
        self.root = dataset_root
        self.name = dataset_name
        self.images_root = os.path.join(dataset_root, "test") # Root directory of images
        self.annotations = os.path.join(dataset_root, "annotations", f"{self.name}.csv") # Root directory of annotations
        self.num_samples_per_class = num_samples_per_class
        self.num_unique_classes = num_unique_classes
        self.seed = seed
        self.pristine = filter_with_pristine
        self.debug = bool(os.getenv("DEBUG", False))
        self.samples = self._load_dataset(num_samples_per_class, num_unique_classes)
        self.classes = list(set([sample[1] for sample in self.samples]))

    def _load_dataset(self, num_samples_per_class, num_unique_classes):
        # Load the dataset
        columns = ["name", "image_path", "leaking_prompt", "support_prompt"]
        df = pd.read_csv(self.annotations)[columns]
        df = df[~df["image_path"].isin(__BROKEN_IMAGES__)]
        df["image_path"] = df["image_path"].apply(lambda x: f"{self.images_root}/{x}")

        if num_unique_classes != -1:
            K = min(num_unique_classes, len(df.name.unique()))
            df = df[df["name"].isin(random.sample(df.name.unique().tolist(), k=K))]
        
        if num_samples_per_class != -1:
            df = df.groupby("name", group_keys=False).apply(lambda x: x.sample(n=min(num_samples_per_class, len(x)), 
                                                                               random_state=self.seed))
            
        if self.pristine is not None:
            pristine_df = pd.read_csv(self.pristine)[["name", "image_path", "leaking"]]
            pristine_df["image_path"] = pristine_df["image_path"].apply(lambda x: f"{self.images_root}/{str('/'.join(x.split('/')[-2:]))}")
            pristine_df = pristine_df.drop_duplicates(subset=["image_path"])
            pristine_df = pristine_df[pristine_df.apply(lambda row: row["name"] in row['leaking'], axis=1)]
            
            new_df = df[df["image_path"].isin(pristine_df["image_path"].tolist())]
            other_df = df[~df["image_path"].isin(pristine_df["image_path"].tolist())]
            non_leaking_support = other_df.sample(frac=float(len(new_df)/len(df)), random_state=self.seed)
            # Support to non leaking images too
            df = pd.concat([new_df, non_leaking_support], ignore_index=True)

        df["image_path"] = df["image_path"].apply(lambda x: x.replace("//", "/"))

        if self.debug:
            return random.sample(df.values.tolist(), k=min(128, len(df)))
        
        
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