"""Portable, lazy-loading command line for the experiments in the TUTUS paper."""
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREE = ('prompting', 'blurring', 'noise', 'pruning')
COMPONENTS = ('vision', 'connector', 'text')
OBJECTIVES = ('dpo', 'sft', 'ga', 'npo')
SPLITS = {'dev': 'TUTUS-DEV', 'celebrity': 'TUTUS-TS-P', 'celebrity-heldout': 'TUTUS-TS-P-HELDOUT', 'fictitious': 'MLLMU-TS-P'}
PAPER = json.loads((ROOT / 'configs/paper.json').read_text())
MODELS = tuple(PAPER['models'])


def dataset_root(a):
    return Path(a.data_root).expanduser().resolve()


def configuration(a):
    from omegaconf import OmegaConf
    model, method = a.model, a.method
    record = PAPER['post_training'].get(f'{model}-{method}-{a.objective}')
    if method in COMPONENTS and record is None:
        raise ValueError(f'No paper configuration for {model}/{method}/{a.objective}')
    root = dataset_root(a)
    output = Path(a.output).expanduser().resolve()
    split = 'fictitious' if a.action == 'transfer' else a.split
    data_kind = 'mllmu' if split == 'fictitious' else 'celebdata'
    ident = f'{model}-{method}' + (f'-{a.objective}' if method in COMPONENTS else '')
    area = 'post-training' if method in COMPONENTS else 'training-free'
    experiment = Path(a.experiment).expanduser().resolve() if a.experiment else output / area / ident
    if split == 'fictitious' and not a.experiment:
        # Anonymization adapters are loaded from post-training; transfer outputs are separate.
        experiment = output / 'post-training' / ident if method in COMPONENTS else output / 'training-free-fictitious' / ident
    cfg = {
        'method': {'name': method},
        'model': {'name': model, 'hf_id': a.model_path},
        'reproducibility': {'seed': a.seed},
        'training': record['training'] if record else None,
        'dataset': {
            'train': {'name': 'TUTUS-PD', 'root': str(root/'celebdata'), 'num_samples_per_class': -1, 'num_unique_classes': -1, 'unique_images': True},
            'test': {'name': SPLITS[split], 'root': str(root/data_kind), 'num_samples_per_class': -1, 'num_unique_classes': -1},
        },
        'evaluation': {'similarity_model': 'sentence-transformers/all-MiniLM-L6-v2', 'pristine_path': str(output/'pristine'), 'jailbreak_prompt': 'Reply only with the name. ', 'test_batch_size_per_gpu': a.batch_size},
        'experiment': {'tag': 'TUTUS', 'path': str(experiment), 'save_path': str(output), 'use_wandb': False, 'wandb_project': 'TUTUS', 'wandb_name': None, 'verbose': False},
    }
    if method in FREE:
        cfg['method'].update(PAPER['training_free'][method])
    if method == 'blurring':
        cfg['method']['facedetection_path'] = str(Path(a.detector).expanduser().resolve())
    if method in COMPONENTS:
        cfg['method'].update(use_lora=True, lora_r=16, lora_alpha=16)
        cfg['training']['num_gpus'] = a.gpus
        cfg['evaluation']['checkpoint'] = a.checkpoint or record['checkpoint']
    if a.action == 'train':
        cfg['dataset']['test']['name'] = SPLITS[a.split]
    if a.action == 'attacks':
        cfg['dataset']['test']['num_unique_classes'] = 100
        cfg['dataset']['test']['num_samples_per_class'] = 2
        cfg['rebuttal'] = {'run_attacks': True, 'run_benign': True, 'compare_pristine': True}
    merged = OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.from_dotlist(a.set_values or []))
    return merged, ident, output


def launch(module, config, a, env, overrides=()):
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(a.gpus), '-m', module, '--config', str(config), *overrides]
    print(shlex.join(cmd), flush=True)
    if not a.dry_run:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_experiment(a):
    from omegaconf import OmegaConf
    cfg, ident, output = configuration(a)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env['TUTUS_DATA_ROOT'] = str(dataset_root(a))
    if a.personalization_adapter:
        env['TUTUS_PERSONALIZATION_ADAPTER'] = str(Path(a.personalization_adapter).expanduser().resolve())
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['WANDB_MODE'] = 'disabled'
    if a.action == 'utility':
        utility(a, cfg, ident, output, env)
        return
    config = output / 'configs' / f'{a.action}-{ident}-{cfg.dataset.test.name}.yaml'
    if not a.dry_run:
        config.parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.experiment.path).mkdir(parents=True, exist_ok=True)
        Path(cfg.evaluation.pristine_path).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config)
    else:
        print(f'# Config: {ident}; split={cfg.dataset.test.name}; output={cfg.experiment.path}')
    if a.action == 'pristine':
        launch('tutus.experiments.cache', config, a, env)
        return
    if not a.dry_run:
        required = Path(cfg.dataset.test.root)/'annotations'/f'{cfg.dataset.test.name}.csv'
        if not required.is_file():
            raise FileNotFoundError(f'Missing {required}. Extract TUTUS-dataset.zip and set --data-root.')
        if a.method == 'blurring' and not Path(cfg.method.facedetection_path).is_file():
            raise FileNotFoundError('Download the face detector with: python run.py download-detector')
        if a.method in COMPONENTS and a.action != 'train':
            checkpoint = Path(cfg.experiment.path)/cfg.evaluation.checkpoint/'mypeft/adapter_model.safetensors'
            if not checkpoint.is_file():
                raise FileNotFoundError(f'Missing anonymization checkpoint: {checkpoint}. Train this configuration first or supply --experiment and --checkpoint.')
        if a.action == 'train' and list(Path(cfg.experiment.path).glob('perc_*/mypeft/adapter_model.safetensors')):
            raise FileExistsError(f'Checkpoints already exist at {cfg.experiment.path}; use a different --output or --experiment for a new training run.')
    cache = Path(cfg.evaluation.pristine_path)/f'{cfg.dataset.test.name}_{cfg.model.name}_pristine.csv'
    # Use full-split base predictions even when a stress test samples a smaller subset.
    if a.dry_run or not cache.is_file():
        launch('tutus.experiments.cache', config, a, env, ('dataset.test.num_unique_classes=-1', 'dataset.test.num_samples_per_class=-1'))
    module = {'train': 'train', 'evaluate': 'evaluate', 'transfer': 'transfer', 'attacks': 'attacks'}[a.action]
    if cfg.dataset.test.name.startswith('MLLMU') and a.action == 'evaluate':
        module = 'transfer'
    launch('tutus.experiments.'+module, config, a, env)


def utility(a, cfg, ident, output, env):
    from omegaconf import OmegaConf
    if a.method == 'pristine':
        cfg.method = {'name': 'pristine'}
    config = output/'configs'/f'utility-{ident}.yaml'
    if not a.dry_run:
        config.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config)
    ids = {'llava_next': 'llava-hf/llava-v1.6-mistral-7b-hf', 'llava_onevision': 'llava-hf/llava-onevision-qwen2-7b-ov-hf', 'llava_onevision_smol': 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf', 'qwen2vl': 'Qwen/Qwen2-VL-7B-Instruct', 'qwen2_5vl': 'Qwen/Qwen2.5-VL-7B-Instruct', 'qwen2_5vl_smol': 'Qwen/Qwen2.5-VL-3B-Instruct'}
    backend = 'qwen2_5_vl' if 'qwen2_5' in a.model else 'qwen2_vl' if 'qwen' in a.model else 'llava_hf'
    checkpoint = cfg.evaluation.get('checkpoint', '')
    opts = f'pretrained={a.model_path or ids[a.model]},custom_config={config},checkpoint_name={checkpoint}'
    cmd = [sys.executable, '-m', 'accelerate.commands.launch', '--num_processes', str(a.gpus), '-m', 'tutus.experiments.utility', '--model', backend, '--model_args', opts, '--tasks', a.benchmarks, '--output_path', str(output/'utility'/ident), '--checkpoint_name', checkpoint]
    if a.limit is not None:
        cmd.extend(['--limit', str(a.limit)])
    print(shlex.join(cmd), flush=True)
    if not a.dry_run:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def personalize(a):
    defaults = PAPER['personalization'][a.model]
    output = Path(a.output).expanduser().resolve()/'personalization'/a.model
    env = os.environ.copy(); env['TUTUS_DATA_ROOT'] = str(dataset_root(a))
    env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH', '')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(a.gpus), '-m', 'tutus.experiments.personalize', '--model_name', a.model, '--save_path', str(output), '--epochs', str(a.epochs or defaults['epochs']), '--learning_rate', str(a.learning_rate or defaults['learning_rate'])]
    print(shlex.join(cmd), flush=True)
    if not a.dry_run:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def check_data(a):
    import pandas as pd
    root = dataset_root(a); errors = []
    for dataset, folder, filenames in [
        ('celebdata', 'test', ['TUTUS-DEV.csv', 'TUTUS-TS-P.csv', 'TUTUS-TS-P-HELDOUT.csv']),
        ('celebdata', 'train', [f'TUTUS-PD-V1-{m}.csv' for m in MODELS]),
        ('mllmu', 'images', ['MLLMU-TS-P.csv']),
    ]:
        for filename in filenames:
            p = root/dataset/'annotations'/filename
            if not p.is_file(): errors.append(str(p)); continue
            df = pd.read_csv(p)
            paths = df.image_path.drop_duplicates().astype(str)
            missing = [x for x in paths if not (root/dataset/folder/x).is_file()]
            print(f'{dataset}/{filename}: {len(df)} rows, {len(paths)} unique images, {len(missing)} missing')
            errors.extend(missing)
    for config in ('ft_Data', 'Full_Set'):
        files = list((root/'mllmu/source'/config).glob('*.parquet'))
        if not files: errors.append(f'Missing personalization source: {config}')
    if errors:
        raise SystemExit('Dataset check failed:\n'+'\n'.join(errors[:20]))
    print('Dataset annotation/image check passed.')


def download_detector(a):
    import gdown
    path = Path(a.detector).expanduser().resolve()
    if path.exists():
        print(f'Using existing detector: {path}'); return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not gdown.download(id='1Ak1Hqtwogd6IzBjmS0xoMW1cqylT8R4K', output=str(path), quiet=False):
        raise RuntimeError('Detector download failed. Download the face detector manually and pass --detector.')


def suite(a):
    commands = []
    for model in a.models:
        if a.suite in ('training-free', 'all'):
            commands += [('evaluate', model, method, 'dpo', []) for method in FREE]
            commands += [('evaluate', model, method, 'dpo', ['--split', 'fictitious']) for method in FREE]
        if a.suite in ('post-training', 'all'):
            for method in COMPONENTS:
                for objective in OBJECTIVES:
                    commands.append(('train', model, method, objective, ['--split', 'dev']))
                    record = PAPER['post_training'][f'{model}-{method}-{objective}']
                    for checkpoint in record['training'].get('save_checkpoints', [10,20,30,40,50,60,80,100]):
                        commands.append(('evaluate', model, method, objective, ['--split', 'dev', '--checkpoint', f'perc_{checkpoint}']))
                    commands.append(('evaluate', model, method, objective, ['--split', 'celebrity']))
        if a.suite in ('transfer', 'all'):
            commands += [('transfer', model, method, objective, []) for method in COMPONENTS for objective in OBJECTIVES]
        if a.suite in ('attacks', 'all'):
            commands.append(('attacks', model, 'vision', 'dpo', []))
        if a.suite in ('noise-ablation', 'pruning-ablation', 'all'):
            methods = ['noise', 'pruning'] if a.suite == 'all' else [a.suite.split('-')[0]]
            for method in methods:
                key = 'level' if method == 'noise' else 'sparsity'
                commands += [('evaluate', model, method, 'dpo', ['--split', 'dev', '--set', f'method.{key}={value}', '--experiment', str(Path(a.output)/'ablations'/f'{model}-{method}-{value}')]) for value in PAPER['ablations'][method]]
        if a.suite in ('utility', 'all'):
            commands += [('utility', model, method, 'dpo', []) for method in ('pristine', *FREE)]
            commands += [('utility', model, method, objective, []) for method in COMPONENTS for objective in OBJECTIVES]
    for action,model,method,objective,extra in commands:
        cmd = [sys.executable, str(ROOT/'run.py'), action, '--model', model, '--method', method, '--objective', objective, '--data-root', str(dataset_root(a)), '--output', a.output, '--gpus', str(a.gpus), *extra]
        print(shlex.join(cmd), flush=True)
        if a.execute: subprocess.run(cmd, cwd=ROOT, check=True)
    print(f'{len(commands)} commands {"executed" if a.execute else "planned; add --execute to run"}.')


def add_common(p):
    p.add_argument('--data-root', default=os.environ.get('TUTUS_DATA_ROOT', str(ROOT/'data/TUTUS-data')))
    p.add_argument('--output', default=str(ROOT/'runs'))
    p.add_argument('--gpus', type=int, default=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('list', help='Show models, methods, objectives, and selected checkpoints')
    p.add_argument('--json', action='store_true')
    p = sub.add_parser('check-data'); add_common(p)
    p = sub.add_parser('download-detector'); p.add_argument('--detector', default=str(ROOT/'assets/yolov8m-face.pt'))
    for action in ('pristine', 'train', 'evaluate', 'attacks', 'transfer', 'utility'):
        p = sub.add_parser(action); add_common(p)
        p.add_argument('--model', choices=MODELS, default='qwen2_5vl')
        p.add_argument('--method', choices=(*FREE, *COMPONENTS, 'pristine'), default='vision' if action in ('train','attacks','transfer') else 'blurring')
        p.add_argument('--objective', choices=OBJECTIVES, default='dpo')
        p.add_argument('--split', choices=tuple(SPLITS), default='dev' if action=='train' else 'celebrity')
        p.add_argument('--checkpoint'); p.add_argument('--experiment'); p.add_argument('--personalization-adapter')
        p.add_argument('--batch-size', type=int, default=4); p.add_argument('--seed', type=int, default=42)
        p.add_argument('--model-path', help='Optional local foundation-model directory')
        p.add_argument('--detector', default=str(ROOT/'assets/yolov8m-face.pt'))
        p.add_argument('--set', dest='set_values', action='append', default=[], help='Repeatable OmegaConf key=value override')
        p.add_argument('--benchmarks', default='mme,mmmu_val,mmbench_en_dev,seedbench,vmcbench')
        p.add_argument('--limit', type=float, help='Optional utility smoke-test sample limit')
        p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('personalize'); add_common(p)
    p.add_argument('--model', choices=MODELS, default='qwen2_5vl'); p.add_argument('--epochs', type=int); p.add_argument('--learning-rate', type=float); p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('suite'); add_common(p)
    p.add_argument('--suite', choices=('training-free','post-training','transfer','attacks','utility','noise-ablation','pruning-ablation','all'), required=True)
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS)); p.add_argument('--execute', action='store_true')
    p = sub.add_parser('report'); p.add_argument('--output', default=str(ROOT/'runs'))
    p = sub.add_parser('score'); p.add_argument('--pristine', required=True); p.add_argument('--predictions', required=True); p.add_argument('--eligible'); p.add_argument('--save')
    a = parser.parse_args()
    if hasattr(a,'gpus') and a.gpus < 1: parser.error('--gpus must be positive')
    if a.action == 'list':
        if a.json: print(json.dumps(PAPER, indent=2))
        else:
            print('Models: '+', '.join(MODELS)); print('Training-free: '+', '.join(FREE)); print('Components: '+', '.join(COMPONENTS)); print('Objectives: '+', '.join(OBJECTIVES))
            for key,row in PAPER['post_training'].items(): print(f'{key}: lr={row["training"]["lr"]}, selected={row["checkpoint"]}')
    elif a.action == 'check-data': check_data(a)
    elif a.action == 'download-detector': download_detector(a)
    elif a.action == 'personalize': personalize(a)
    elif a.action == 'suite': suite(a)
    elif a.action == 'report':
        from tutus.report import report
        report(Path(a.output).expanduser().resolve())
    elif a.action == 'score':
        from tutus.score import load, score
        result = score(load(a.pristine), load(a.predictions), json.loads(Path(a.eligible).read_text()) if a.eligible else None)
        text = json.dumps(result, indent=2)+'\n'
        if a.save:
            p=Path(a.save); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text)
        print(text, end='')
    else:
        if a.action == 'train' and a.method not in COMPONENTS: parser.error('train requires --method vision, connector, or text')
        if a.action == 'train' and a.split == 'fictitious': parser.error('Use personalize for fictitious-identity learning; anonymization training uses TUTUS-Train.')
        run_experiment(a)
