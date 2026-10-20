from tutus.utils.dist import is_main_process, get_rank
from datetime import datetime
from omegaconf import OmegaConf
import os
import json
import wandb
from rich import print

class Logger:
    def __init__(self, experiment_path, run_id, use_wandb, wandb_name=None, wandb_project=None, configs=None):
        self.experiment_path = experiment_path
        self.run_id = run_id
        self.use_wandb = use_wandb
        self.name = wandb_name
        self.project = wandb_project
        self.configs = configs
        self.metrics_name = "metrics.json"
        self.metrics_tracker = {}

        if self.use_wandb and is_main_process():
            self.run = self.setup_wandb(self.project, self.name, OmegaConf.to_container(configs))


    def setup_wandb(self, project, name, configs):
        run = wandb.init(project=project, name=name, config=configs)
        return run
    
    def terminate_experiment(self):
        if is_main_process() and self.use_wandb:
            self.run.finish()

    def log_wandb(self, data):
        if is_main_process():
            self.run.log(data)

    def log_metrics(self, data):
        for key, value in data.items():
            if key not in self.metrics_tracker:
                self.metrics_tracker[key] = []
            self.metrics_tracker[key].append(value)
        
        if self.use_wandb:
            self.log_wandb(data)
    
    def save_metrics(self):
        if is_main_process():
            self.save_json(self.metrics_tracker, self.metrics_name)
    
    @staticmethod
    def info(data, status="INFO"):
        if is_main_process():
            # Write datetime in the format HH:MM:SS
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}|[bold magenta]{status}[/bold magenta]] {data}", flush=True)

    def save_json(self, data, filename):
        if is_main_process():
            file_path = os.path.join(self.experiment_path, filename)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            self.info(f"Saved {filename} to {self.experiment_path}/{filename}", status="SAVING")

    def load_json(self, filename):
        file_path = os.path.join(self.experiment_path, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def log_model(self, data):
        pass
