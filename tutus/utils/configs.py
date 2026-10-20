class MethodFactory:
    @staticmethod
    def get_method(config):
        classes = {"pristine": PristineConfig, "prompting": PromptingConfig,
                   "noise": NoiseConfig, "pruning": PruningConfig, "blurring": BlurringConfig,
                   "vision": VisionConfig, "connector": ConnectorConfig, "text": TextConfig}
        try:
            return classes[config.get("name")](config)
        except KeyError as exc:
            raise ValueError(f"Unknown method: {config.get('name')}") from exc

class TrainFactory:

    @staticmethod
    def get_train_config(config):
        name = config.loss.get('name')
        if name == 'dpo':
            return DPOTrainConfig(config)
        elif name == 'sft':
            return SFTTrainConfig(config)
        elif name == 'ga':
            return GAConfig(config)
        elif name == 'npo':
            return NPOConfig(config)
        else:
            raise ValueError(f'Unknown training name: {name}')

class DatasetFactory:

    @staticmethod
    def get_dataset_config(config):
        return DatasetConfig(config)

class Config:

    def __str__(self):
        label = ''
        for attr, value in self.__dict__.items():
            label += f'{attr} --> {value}\n'
        return label[:-1]

class DatasetConfig(Config):

    def __init__(self, config):
        self.name = config.get('name')

class TrainConfig(Config):

    def __init__(self, config):
        self.train_batch_size_per_gpu = config.get('train_batch_size_per_gpu', 1)
        self.test_batch_size_per_gpu = config.get('batch_size_per_gpu', 4)
        self.total_batch_size = config.get('total_batch_size', 1)
        self.num_gpus = config.get('num_gpus', 8)
        self.use_early_stopping = config.get('use_early_stopping', False)
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        self.num_epochs = config.get('num_epochs', 1)
        self.save_checkpoints = config.get('save_checkpoints', [10, 20, 30, 40, 50, 60, 80, 100])
        self.optimizer = config.get('optimizer', 'adamw')
        self.lr = config.get('lr', 0.001)
        self.weight_decay = config.get('weight_decay', 0)
        self.lr_scheduler = config.get('lr_scheduler', 'cosine')
        self.warmup_ratio = config.get('warmup_ratio', 0.1)
        self.gradient_checkpointing = config.get('gradient_checkpointing', True)
        if self.num_gpus < 1 or self.train_batch_size_per_gpu < 1:
            raise ValueError('GPU count and per-GPU batch size must be positive')
        if self.total_batch_size % (self.train_batch_size_per_gpu * self.num_gpus):
            raise ValueError('Total batch size must be divisible by the distributed microbatch size')
        self.accumulation_steps = self.total_batch_size // (self.train_batch_size_per_gpu * self.num_gpus)
        if self.accumulation_steps < 1:
            raise ValueError('Total batch size must be at least the distributed microbatch size')
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

class MethodConfig(Config):

    def __init__(self, config):
        self.name = config.get('name')
        self.training_free = config.get('training_free')
        self.requires_perturbation = config.get('requires_perturbation')

class TrainableMethodConfig(MethodConfig):

    def __init__(self, config):
        super().__init__(config)
        self.training_free = False
        self.use_lora = config.get('use_lora', True)

class TrainingFreeMethodConfig(MethodConfig):

    def __init__(self, config):
        super().__init__(config)
        self.training_free = True

class DPOTrainConfig(TrainConfig):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'dpo'
        self.beta = config.get('beta', 0.1)
        self.use_ref = True
        self.use_support = False

class SFTTrainConfig(TrainConfig):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'sft'
        self.use_ref = False
        self.use_support = False

class GAConfig(TrainConfig):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'ga'
        self.use_ref = False
        self.use_support = False

class NPOConfig(TrainConfig):

    def __init__(self, config):
        super().__init__(config)
        self.name = 'npo'
        self.beta = config.get('beta', 0.1)
        self.use_ref = True
        self.use_support = False

class PristineConfig(TrainingFreeMethodConfig):

    def __init__(self, config):
        super().__init__(config)

class PromptingConfig(TrainingFreeMethodConfig):

    def __init__(self, config):
        super().__init__(config)
        self.system_prompt = config.get('system_prompt', 'Do not reveal the identity of the person in the image.')

class NoiseConfig(TrainingFreeMethodConfig):

    def __init__(self, config):
        config.setdefault('requires_perturbation', True)
        super().__init__(config)
        self.level = config.get('level', 2.0)
        self.position = config.get('position', 'pre_connector')

class PruningConfig(TrainingFreeMethodConfig):

    def __init__(self, config):
        config.setdefault('requires_perturbation', True)
        super().__init__(config)
        self.sparsity = config.get('sparsity', 0.7)
        self.position = config.get('position', 'pre_connector')

class BlurringConfig(TrainingFreeMethodConfig):

    def __init__(self, config):
        super().__init__(config)
        self.facedetection_path = config.get('facedetection_path', './assets/yolov8m-face.pt')

class ConnectorConfig(TrainableMethodConfig):

    def __init__(self, config):
        config.setdefault('use_lora', True)
        config.setdefault('requires_perturbation', False)
        super().__init__(config)
        self.lora_r = config.get('lora_r', 16)
        self.lora_alpha = config.get('lora_alpha', 32)
        self.lora_bias = config.get('lora_bias', None)

class VisionConfig(TrainableMethodConfig):

    def __init__(self, config):
        config.setdefault('use_lora', True)
        config.setdefault('requires_perturbation', False)
        super().__init__(config)
        self.lora_r = config.get('lora_r', 16)
        self.lora_alpha = config.get('lora_alpha', 32)
        self.lora_bias = config.get('lora_bias', None)

class TextConfig(TrainableMethodConfig):

    def __init__(self, config):
        config.setdefault('use_lora', True)
        config.setdefault('requires_perturbation', False)
        super().__init__(config)
        self.lora_r = config.get('lora_r', 16)
        self.lora_alpha = config.get('lora_alpha', 32)
        self.lora_bias = config.get('lora_bias', None)

class CelebDataTrainConfig(DatasetConfig):

    def __init__(self, config):
        super().__init__(config)
        self.root = config.get('root')
        self.num_samples_per_class = config.get('num_samples_per_class', -1)
        self.num_unique_classes = config.get('num_unique_classes', -1)
        self.unique_images = config.get('unique_images', True)

class CelebDataTestConfig(DatasetConfig):

    def __init__(self, config):
        super().__init__(config)
        self.root = config.get('root')
        self.num_samples_per_class = config.get('num_samples_per_class', -1)
        self.num_unique_classes = config.get('num_unique_classes', -1)

class MLLMUBenchConfig(DatasetConfig):

    def __init__(self, config):
        super().__init__(config)
        self.root = config.get('root')
        self.split = config.get('split', 'test')
        self.batch_size = config.get('batch_size', 1)
        self.num_samples = config.get('num_samples', -1)
        self.subset = config.get('subset', None)
        self.seed = config.get('seed', 42)
