from tutus.evaluation.summary import harmonic_mean
import os
import json
import torch

import pandas as pd

from omegaconf import OmegaConf
from tutus.utils import setup_reproducibility

from transformers import get_cosine_schedule_with_warmup
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from tutus.utils.dist import (setup_distributed, 
                        cleanup_distributed, 
                        use_distributed, 
                        create_global_id,
                        get_device_id,
                        is_main_process)

from tutus.utils.const import __MODEL_ZOO_MLLMU__

from tutus.utils import (count_trainable_parameters,
                   Logger,
                   Color, 
                   load_config, 
                   freeze_model,
                   setup_reproducibility)


from tutus.dataset import (load_train_dataset, 
                     load_test_dataset, 
                     load_test_dataloader,
                     load_train_valid_dataloader)

from tutus.training import (train_model,
                      EarlyStopping,
                      load_optimizer,
                      ResizeLongSideMax)

from tutus.evaluation import (contextual_description_similarity, 
                        anonymization_success_rate,
                        jailbreak_defense_rate)

from tutus.methods import (generate_pristine, 
                     generate_prompting, 
                     generate_noise,
                     generate_pruning,
                     generate_blurring,
                     generate_tuned)

from tutus.utils.const import (__MODEL_ZOO__, 
                         __METHODS__, 
                         __TRAINING_FREE__,
                         __LOSSES__,
                         __METHODS_WITH_TOKENS_PERTURBATION__,
                         __CONNECTOR_MODULE__,
                         __VISION_MODULE__,
                         __LANGUAGE_MODULE__)


from tutus.modeling import (load_model, 
                      load_perturbation_layer, 
                      load_lora,
                      setup_lora)

from tutus.utils.configs import MethodFactory, TrainFactory


class ResizeLongSideMax:
    def __init__(self, max_long):
        self.max_long = max_long

    def __call__(self, img):
        w, h = img.size
        long = max(w, h)

        if long <= self.max_long:
            return img  # No resizing needed

        # Scale so the longest side becomes max_long
        scale = self.max_long / long
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        return transforms.Resize((new_h, new_w))(img)


def main(cfg):

    if not "path" in cfg.experiment or cfg.experiment.path is None:
        exp_path = cfg.experiment.save_path+"/"+cfg.experiment.tag+"/"+cfg.model.name+"/"+cfg.method.name+"/"+cfg.experiment.run_id
        cfg.experiment.path = exp_path
        os.makedirs(exp_path, exist_ok=True)
        print(f"Experiment path not specified. Setting to {exp_path}")
    else:
        exp_path = cfg.experiment.path

    cfg_method = MethodFactory.get_method(cfg.method)  

    # Setup reproducibility
    setup_reproducibility(cfg.reproducibility.seed)
    
    # Create logger
    logger = Logger(cfg.experiment.path, 
                    cfg.experiment.run_id, 
                    use_wandb=False,
                    configs=cfg)

    # Log parameters for the experiment
    logger.info("Experiment params {}".format(OmegaConf.to_yaml(cfg)))

    cfg.evaluation.pristine_path = os.path.join(cfg.evaluation.pristine_path, f"{cfg.dataset.test.name}_{cfg.model.name}_pristine.csv")
    
    
    # Close experiment if pristine_only is set
    if not os.path.exists(cfg.evaluation.pristine_path):
        # Load dataset and dataloader (with distributed sampler if applicable)
        test_dataset = load_test_dataset(cfg.dataset.test, 
                                         pristine_path=None, 
                                         seed=cfg.reproducibility.seed)
        test_dataloader = load_test_dataloader(test_dataset, 
                                               cfg.evaluation.test_batch_size_per_gpu)
        logger.info("Dataset loaded successfully.")

        if cfg.experiment.verbose: logger.info("Dataset: {}".format(test_dataset))
        # If needed, generate pristine predictions, needed for anonymization success rate (AnSR) and contextual description preservation (CDP)
        
        model, processor = load_model(cfg.model, 
                                      cfg_method, 
                                      pristine=True)
        # Load peft 
        model = load_lora(model, __MODEL_ZOO_MLLMU__[cfg.model.name], "mypeft")
        model = model.merge_and_unload()
        model = model.to(get_device_id())
        check = generate_pristine(model, processor, test_dataloader, cfg.evaluation.pristine_path)
        logger.info(f"{Color.magenta('PRISTINE')} predictions generated.")


        logger.terminate_experiment()
        if use_distributed(): cleanup_distributed()
        logger.info("Experiment finished. Re-Run the script to continue training.")
        exit()
    
    else:
        test_dataset = load_test_dataset(cfg.dataset.test, 
                                         pristine_path=cfg.evaluation.pristine_path, 
                                         seed=cfg.reproducibility.seed)
        test_dataloader = load_test_dataloader(test_dataset, 
                                               cfg.evaluation.test_batch_size_per_gpu)
        logger.info("Dataset loaded successfully.")

        if cfg.experiment.verbose: logger.info("Dataset: {}".format(test_dataset))

    # If requires training, load the training dataset 
    # Check if preference dataset is already generated and eventually generate it
    
    # Load model and processor
    model, processor = load_model(cfg.model, 
                                  cfg_method)
    # Load peft 
    model = load_lora(model, __MODEL_ZOO_MLLMU__[cfg.model.name], "mypeft")
    model = model.merge_and_unload()
    model = model.to(get_device_id())

    logger.info("Model loaded successfully.")
    if cfg.experiment.verbose: logger.info("Model: {}".format(model))


    if cfg_method.training_free: 

        if cfg_method.name == "prompting":
            anonymized_path = os.path.join(cfg.experiment.path, "prompting.csv")
            check = generate_prompting(model, processor, test_dataloader, anonymized_path, cfg_method.system_prompt, cfg.evaluation.jailbreak_prompt)

        elif cfg_method.name == "noise":
            anonymized_path = os.path.join(cfg.experiment.path, "noise.csv")
            check = generate_noise(model, processor, test_dataloader, anonymized_path, cfg.evaluation.jailbreak_prompt)

        elif cfg_method.name == "pruning":
            anonymized_path = os.path.join(cfg.experiment.path, "pruning.csv")
            check = generate_pruning(model, processor, test_dataloader, anonymized_path, cfg.evaluation.jailbreak_prompt)

        elif cfg_method.name == "blurring":
            anonymized_path = os.path.join(cfg.experiment.path, "blurring.csv")
            check = generate_blurring(model, processor, test_dataloader, anonymized_path, cfg_method.facedetection_path, cfg.evaluation.jailbreak_prompt)

        else:
            raise ValueError(f"Unknown training free method: {cfg_method.name}")
        
        logger.metrics_name = cfg.experiment.path + "___" + cfg.dataset.test.name + "___metrics.json"
        base_classes = None

    else:
        ############################################################
        ########## Setting up the training configurations ##########      
        ############################################################

        cfg_training = TrainFactory.get_train_config(cfg.training)

        # As for now no base_classes for MLLMU-Bench
        base_classes = None
        
        if "llava" in cfg.model.name:
            resizer = ResizeLongSideMax(512)
        else:
            resizer = None

        
        if cfg_method.requires_perturbation:
            model.load_pretrained(cfg.experiment.path + "/" + cfg.evaluation.checkpoint)
        elif cfg_method.use_lora:
            model = load_lora(model, cfg.experiment.path + "/" + cfg.evaluation.checkpoint, "mypeft")
            model = model.merge_and_unload()
        else:
            model.load_state_dict(torch.load(os.path.join(cfg.experiment.path + "/" + cfg.evaluation.checkpoint, "parameters_trained.pt"), map_location="cpu"))

        model.eval()

        anonymized_path = os.path.join(cfg.experiment.path, cfg.evaluation.checkpoint, f"{cfg.dataset.test.name}___{cfg_method.name}.csv")

        if os.path.exists(anonymized_path):
            logger.info(f"Anonymized file {anonymized_path} already exists. Skipping generation.")
        else:
            check = generate_tuned(model, 
                                  processor, 
                                  test_dataloader, 
                                  cfg_method.name, 
                                  anonymized_path, 
                                  cfg.evaluation.jailbreak_prompt)
        
        # Will be save in the checkpoint folder hopefully
        logger.metrics_name = cfg.evaluation.checkpoint + "___" + cfg.dataset.test.name + "___metrics.json"

    # Evaluation
    if is_main_process():
        pristine = pd.read_csv(cfg.evaluation.pristine_path)
        anonymized = pd.read_csv(anonymized_path)

        # Compute ASR
        total_asr, base_asr, novel_asr = anonymization_success_rate(pristine, anonymized, base=base_classes)
        logger.info(f"Anonymization Success Rate (ASR): {total_asr*100}% | Base ASR: {base_asr*100}% | Novel ASR: {novel_asr*100}%")
        logger.log_metrics({"average_asr": total_asr, "base_asr": base_asr, "novel_asr": novel_asr})

        # Compute CDS
        total_cds, base_cds, novel_cds = contextual_description_similarity(pristine, anonymized, cfg.model.name,
                                                                            similarity_model=cfg.evaluation.similarity_model, base=base_classes)
        logger.info(f"Contextual Description Similarity (CDP): {total_cds*100}% | Base CDP: {base_cds*100}% | Novel CDP: {novel_cds*100}%")
        logger.log_metrics({"average_cdp": total_cds, "base_cdp": base_cds, "novel_cdp": novel_cds})

        total_jdr, base_jdr, novel_jdr = jailbreak_defense_rate(pristine, anonymized, total_asr, base_asr, novel_asr, base=base_classes)
        logger.info(f"Jailbreak Defense Rate (JDR): {total_jdr*100}% | Base JDR: {base_jdr*100}% | Novel JDR: {novel_jdr*100}%")
        logger.log_metrics({"average_jdr": total_jdr, "base_jdr": base_jdr, "novel_jdr": novel_jdr})

        # Log f1 score
        denom_total = (total_asr + total_cds)
        denom_base = (base_asr + base_cds)
        denom_novel = (novel_asr + novel_cds)
        
        if denom_total > 0:
            average_f1 = round(2*(total_asr * total_cds) / denom_total, 4)
        else:
            average_f1 = 0.0
        if denom_base > 0:
            base_f1 = round(2*(base_asr * base_cds) / denom_base, 4)
        else:
            base_f1 = 0.0
        if denom_novel > 0:
            novel_f1 = round(2*(novel_asr * novel_cds) / denom_novel, 4)
        else:
            novel_f1 = 0.0 

        logger.info(f"Average F1: {average_f1*100}% | Base F1: {base_f1*100}% | Novel F1: {novel_f1*100}%")
        logger.log_metrics({"average_f1": average_f1, "base_f1": base_f1, "novel_f1": novel_f1})

        # Log f1 score
        denom_total = (total_asr + total_cds + total_jdr)
        denom_base = (base_asr + base_cds + base_jdr)
        denom_novel = (novel_asr + novel_cds + novel_jdr)
        
        if denom_total > 0:
            average_hm = round(harmonic_mean(total_asr, total_cds, total_jdr), 4)
        else:
            average_hm = 0.0
        if denom_base > 0:
            base_hm = round(harmonic_mean(base_asr, base_cds, base_jdr), 4)
        else:
            base_hm = 0.0
        if denom_novel > 0:
            novel_hm = round(harmonic_mean(novel_asr, novel_cds, novel_jdr), 4)
        else:
            novel_hm = 0.0

        logger.info(f"Average HM: {average_hm*100}% | Base HM: {base_hm*100}% | Novel HM: {novel_hm*100}%")
        logger.log_metrics({"average_hm": average_hm, "base_hm": base_hm, "novel_hm": novel_hm})
        
        logger.save_metrics()
    
    logger.terminate_experiment()
    if use_distributed(): cleanup_distributed()
    
    logger.info("Experiment finished.")
    

if __name__ == "__main__":
    
    cfg = load_config()
    # Bootstrap the experiment
    # Setup distributed training if applicable
    if use_distributed(): setup_distributed()
    
    # Set unique identifier for the experiment
    # even in a distributed setting
    run_id = create_global_id()
    cfg.experiment.run_id = run_id.__str__()

    main(cfg)
    
