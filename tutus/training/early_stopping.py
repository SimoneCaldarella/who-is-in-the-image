from tutus.modeling import generate_predictions
from tutus.utils.dist import barrier, use_distributed, gather_dataframes_across_ranks, get_rank, get_device_id
import torch.distributed as tdist
from tutus.evaluation.metrics import contextual_description_similarity, anonymization_success_rate
import pandas as pd
import torch
from contextlib import nullcontext
from tutus.utils import Logger
from tutus.utils.dist import is_main_process

__LIMIT__ = 0.8

@torch.no_grad()
def run_validation(model, processor, valid, method, model_name):

    model.eval()
    columns = ["name", "image_path", "support", "leaking"]

    sample = {k: [] for k in columns}
    anonymized_df = pd.DataFrame(sample)
    pristine_df = pd.DataFrame(sample)
    
    s_prompt = "Describe the picture."
    l_prompt = "Who is the celebrity in the picture?"
    pp = False

    for batch in valid:
        _, names, image_paths, _, _, _, _ = batch
        support_prompts = [{"question": s_prompt} for i in range(len(image_paths))]
        leaking_prompts = [{"question": l_prompt} for i in range(len(image_paths))]

        anonymized_support_outputs = generate_predictions(model, processor, support_prompts, image_paths, use_distributed=use_distributed())
        anonymized_leaking_outputs = generate_predictions(model, processor, leaking_prompts, image_paths, use_distributed=use_distributed())

        context = None

        if use_distributed():
            if method == "camouflage":
                model.module.anonymize(False)
            else:
                context = model.module.disable_adapter
        else:
            if method == "camouflage":
                model.anonymize(False)
            else:
                context = model.disable_adapter
        
        with context() if context else nullcontext():
        
            with torch.no_grad():
                pristine_support_outputs = generate_predictions(model, 
                                                                processor, 
                                                                support_prompts, 
                                                                image_paths, 
                                                                use_distributed=use_distributed())
                pristine_leaking_outputs = generate_predictions(model,
                                                                processor, 
                                                                leaking_prompts, 
                                                                image_paths, 
                                                                use_distributed=use_distributed())

        if method == "camouflage":
            if use_distributed: model.module.anonymize(True)
            else: model.module.anonymize(True)


        new_predictions_anonymized = pd.DataFrame({"name": names,
                                                   "image_path": image_paths,
                                                   "support": anonymized_support_outputs,
                                                   "leaking": anonymized_leaking_outputs})
        new_predictions_pristine = pd.DataFrame({"name": names,
                                                 "image_path": image_paths,
                                                 "support": pristine_support_outputs,
                                                 "leaking": pristine_leaking_outputs})
        
        if not pp:
            Logger.info(f"Generated {anonymized_support_outputs[0]}")
            Logger.info(f"Correct output was: {pristine_support_outputs[0]}")
            if is_main_process(): print("---------------")
            pp = True
        
        anonymized_df = pd.concat([anonymized_df, new_predictions_anonymized], ignore_index=True)
        pristine_df = pd.concat([pristine_df, new_predictions_pristine], ignore_index=True)

    anonymized_df_all = gather_dataframes_across_ranks(anonymized_df)
    pristine_df_all = gather_dataframes_across_ranks(pristine_df)

    barrier()

    if get_rank() == 0:
        # Merge all DataFrames
        anonymized_df = pd.concat(anonymized_df_all, ignore_index=True)
        pristine_df = pd.concat(pristine_df_all, ignore_index=True)

        ansr, _, _ = anonymization_success_rate(pristine_df, anonymized_df, base=None, disable=True)
        Logger.info(f"Anonymization Success Rate (ASR): {ansr}")
        cds, _, _ = contextual_description_similarity(pristine_df, anonymized_df, model_name=model_name, similarity_model="sentence-transformers/all-MiniLM-L6-v2", base=None, disable=True)
    else:
        ansr = 0
        cds = 0

    model.train()

    return cds


class EarlyStopping:

    def __init__(self, patience, enabled, valid_dl, method, model_name, logger):
        self.patience = patience
        self.enabled = enabled
        self.valid_dl = valid_dl
        self.method = method
        self.every = 1 # steps
        self.count = 0
        self.count_no_improvement = 0
        self.model_name = model_name
        self.logger = logger

    def check(self, model, processor):

        if self.enabled:
            save_model = False
            stop_training = False

            self.count += 1
            flag_tensor = torch.tensor([0], device=get_device_id())
            if self.count % self.every == 0:
                self.logger.info(f"Running microvalidation for {self.count}th step")
                cds = run_validation(model, processor, self.valid_dl, self.method, self.model_name)
                
                if get_rank() == 0:
                    self.logger.info(f"CDS: {cds}")
            
                    if cds < __LIMIT__:
                        flag_tensor[0] = 1  # True condition
                
            tdist.broadcast(flag_tensor, src=0)


            if flag_tensor.item() == 1:
                self.count_no_improvement += 1
                save_model = True
                if self.count_no_improvement == self.patience:
                    stop_training = True

            else:
                self.count_no_improvement = 0

        
            return save_model, stop_training
        else:
            return False, False

                 

