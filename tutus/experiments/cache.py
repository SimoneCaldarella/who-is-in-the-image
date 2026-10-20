"""Generate the pristine prediction cache used by all relative metrics."""
import os
from pathlib import Path
from tutus.dataset import load_test_dataset, load_test_dataloader
from tutus.methods import generate_pristine
from tutus.modeling import load_model, load_lora
from tutus.utils import load_config, setup_reproducibility
from tutus.utils.configs import MethodFactory
from tutus.utils.const import __MODEL_ZOO_MLLMU__
from tutus.utils.dist import use_distributed, setup_distributed, cleanup_distributed, get_device_id


def main():
    cfg=load_config()
    distributed=use_distributed()
    if distributed: setup_distributed()
    try:
        setup_reproducibility(cfg.reproducibility.seed)
        path=Path(cfg.evaluation.pristine_path)/f'{cfg.dataset.test.name}_{cfg.model.name}_pristine.csv'
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_file():return
        dataset=load_test_dataset(cfg.dataset.test, pristine_path=None, seed=cfg.reproducibility.seed)
        loader=load_test_dataloader(dataset,cfg.evaluation.test_batch_size_per_gpu)
        model,processor=load_model(cfg.model,MethodFactory.get_method({'name':'pristine'}),pristine=True)
        if cfg.dataset.test.name.startswith('MLLMU'):
            model=load_lora(model,__MODEL_ZOO_MLLMU__[cfg.model.name],'mypeft')
            model=model.merge_and_unload()
        model=model.to(get_device_id())
        generate_pristine(model,processor,loader,str(path))
    finally:
        if distributed:cleanup_distributed()

if __name__=='__main__':main()
