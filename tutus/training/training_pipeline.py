import torch
from tutus.utils.dist import get_device_id, use_distributed
from tutus.utils import Logger, progress, Color
import os
from tutus.training.logprobs import compute_logprobs
from tutus.training.losses import dpo, sft, unlearn, npo
from PIL import Image
from tutus.utils import conversation, count_trainable_parameters
from tutus.utils.dist import barrier, is_main_process
from contextlib import nullcontext
from tutus.modeling.lora_tune import save_lora
from torch.distributed import barrier
from tutus.utils.const import __GEN_TOKEN__, __DESCRIBE_PICTURE_PARAPHRASES__
import random



def model_step(model, processor, batch, transform, cfg_training, cfg_method, step_ratio):
    model_device = model.device if hasattr(model, "device") else model.module.device
    model_dtype = model.dtype if hasattr(model, "dtype") else model.module.dtype

    idx, name, image_path, question_text, chosen_text, rejected_text, support_text = batch
    images = []
    chosen_conv_templs = []
    rejected_conv_templs = []
    support_conv_templs = []
    chosen_ans_templs = []
    rejected_ans_templs = []
    support_ans_templ = []

    transform = transform if transform is not None else lambda x: x
    

    for i in range(len(image_path)):
        images.append([transform(Image.open(image_path[i]).convert("RGB"))])

        support_conv = conversation(question=random.choice(__DESCRIBE_PICTURE_PARAPHRASES__), answer=support_text[i])
        support_conv = processor.apply_chat_template(support_conv)
        support_conv_templs.append(support_conv)

        chosen_conv = conversation(question=question_text[i], answer=chosen_text[i])
        chosen_conv = processor.apply_chat_template(chosen_conv)
        chosen_conv_templs.append(chosen_conv)

        rejected_conv = conversation(question=question_text[i], answer=rejected_text[i])
        rejected_conv = processor.apply_chat_template(rejected_conv)
        rejected_conv_templs.append(rejected_conv)

        chosen_ans = conversation(answer=chosen_text[i])
        chosen_ans = processor.apply_chat_template(chosen_ans)
        chosen_ans = chosen_ans.split(__GEN_TOKEN__[(model.module if hasattr(model, "module") else model).custom_name])[-1]
        chosen_ans_templs.append(chosen_ans)

        rejected_ans = conversation(answer=rejected_text[i])
        rejected_ans = processor.apply_chat_template(rejected_ans)
        rejected_ans = rejected_ans.split(__GEN_TOKEN__[(model.module if hasattr(model, "module") else model).custom_name])[-1]
        rejected_ans_templs.append(rejected_ans)

        support_ans = conversation(answer=support_text[i])
        support_ans = processor.apply_chat_template(support_ans)
        support_ans = support_ans.split(__GEN_TOKEN__[(model.module if hasattr(model, "module") else model).custom_name])[-1]
        support_ans_templ.append(support_ans)


    chosen_ans_tokens = processor(text=chosen_ans_templs, return_tensors="pt", padding=True).input_ids
    rejected_ans_tokens = processor(text=rejected_ans_templs, return_tensors="pt", padding=True).input_ids

    chosen_conv_tokens = processor(images=images, text=chosen_conv_templs, return_tensors="pt", padding=True).to(model_device).to(model_dtype)
    rejected_conv_tokens = processor(images=images, text=rejected_conv_templs, return_tensors="pt", padding=True).to(model_device).to(model_dtype)

    if cfg_training.use_support:
        support_conv_tokens = processor(images=images, text=support_conv_templs, return_tensors="pt", padding=True).to(model_device).to(model_dtype)
        support_ans_tokens = processor(text=support_ans_templ, return_tensors="pt", padding=True).input_ids
        support_mask = torch.zeros_like(support_conv_tokens.input_ids)

    chosen_mask = torch.zeros_like(chosen_conv_tokens.input_ids)
    rejected_mask = torch.zeros_like(rejected_conv_tokens.input_ids)

    for i in range(chosen_conv_tokens.input_ids.shape[0]):
        chosen_mask[i, -chosen_ans_tokens[i].shape[0] :] = 1
        rejected_mask[i, -rejected_ans_tokens[i].shape[0] :] = 1
        if cfg_training.use_support:
            support_mask[i, -support_ans_tokens[i].shape[0] :] = 1

    chosen_mask = chosen_mask.bool()
    rejected_mask = rejected_mask.bool()
    if cfg_training.use_support:
        support_mask = support_mask.bool()

    chosen_labels = chosen_conv_tokens.input_ids.clone()
    rejected_labels = rejected_conv_tokens.input_ids.clone()
    if cfg_training.use_support:
        support_labels = support_conv_tokens.input_ids.clone()

    if cfg_training.use_ref:

        model.eval()
        context = None
        
        if use_distributed():
            if cfg_method.requires_perturbation:
                model.module.anonymize(False)
            else:
                context = model.module.disable_adapter
        else:
            if cfg_method.requires_perturbation:
                model.anonymize(False)
            else:
                context = model.disable_adapter
        
        with context() if context else nullcontext():
        
            with torch.no_grad():
            
                ref_chosen_outputs = model(**chosen_conv_tokens)
                ref_rejected_outputs = model(**rejected_conv_tokens)

                if cfg_training.use_support:
                    ref_support_outputs = model(**support_conv_tokens)


            ref_chosen_logprobs = compute_logprobs(ref_chosen_outputs.logits, chosen_labels, selection_mask=chosen_mask)
            ref_rejected_logprobs = compute_logprobs(ref_rejected_outputs.logits, rejected_labels, selection_mask=rejected_mask)
            
            if cfg_training.use_support:
                ref_support_logprobs = compute_logprobs(ref_support_outputs.logits, support_labels, selection_mask=support_mask)

        if cfg_method.requires_perturbation:
            if use_distributed(): model.module.anonymize(True)
            else: model.module.anonymize(True)
        
        model.train()

    pol_chosen_logprobs = None
    pol_rejected_logprobs = None

    if cfg_training.name == "dpo":
        pol_chosen_outputs = model(**chosen_conv_tokens)
        if cfg_method.name == "adapter":
            pass
            # rec_loss_chosen = model.module.perturbation_layer._current_recon_loss.clone()
            # rej_gate_value = model.module.perturbation_layer._current_gate_value.clone()
        pol_chosen_logprobs = compute_logprobs(pol_chosen_outputs.logits, chosen_labels, selection_mask=chosen_mask)
        
        
        pol_rejected_outputs = model(**rejected_conv_tokens)
        if cfg_method.name == "adapter":
            pass
            # rec_loss_rejected = model.module.perturbation_layer._current_recon_loss.clone()
            # chos_gate_value = model.module.perturbation_layer._current_gate_value.clone()
        pol_rejected_logprobs = compute_logprobs(pol_rejected_outputs.logits, rejected_labels, selection_mask=rejected_mask)


        dpo_loss, chosen_rewards, rejected_rewards = dpo(pol_chosen_logprobs,
                                                         pol_rejected_logprobs,
                                                         ref_chosen_logprobs,
                                                         ref_rejected_logprobs,
                                                         beta=cfg_training.beta)
        
        if not cfg_method.name == "adapter":
            return {"dpo_loss": dpo_loss, 
                    "chosen_rewards": chosen_rewards, 
                    "rejected_rewards": rejected_rewards}
        else:
            # dpo_loss = max(step_ratio, 0.2)*dpo_loss + min(0.8, 1-step_ratio)*(rec_loss_chosen + rec_loss_rejected)/2
            # wd = (chos_gate_value.mean()**2 + rej_gate_value.mean()**2)/2
            # dpo_loss = dpo_loss + 0.1 * wd
            return {"dpo_loss": dpo_loss, 
                    "chosen_rewards": chosen_rewards, 
                    "rejected_rewards": rejected_rewards}

    elif cfg_training.name == "dpo_plus":
        pol_chosen_outputs = model(**chosen_conv_tokens)
        pol_chosen_logprobs = compute_logprobs(pol_chosen_outputs.logits, chosen_labels, selection_mask=chosen_mask)

        pol_rejected_outputs = model(**rejected_conv_tokens)
        pol_rejected_logprobs = compute_logprobs(pol_rejected_outputs.logits, rejected_labels, selection_mask=rejected_mask)

        pol_support_outputs = model(**support_conv_tokens)
        pol_support_logprobs = compute_logprobs(pol_support_outputs.logits, support_conv_tokens.input_ids.clone())

        dpo_loss, chosen_rewards, rejected_rewards = dpo(pol_chosen_logprobs,
                                                         pol_rejected_logprobs,
                                                         ref_chosen_logprobs,
                                                         ref_rejected_logprobs,
                                                         beta=cfg_training.beta)
        
        support_loss = sft(pol_support_logprobs)
        dpo_loss += cfg_training.alpha * support_loss

        return {"dpo_loss": dpo_loss, 
                "support_loss": support_loss,
                "chosen_rewards": chosen_rewards, 
                "rejected_rewards": rejected_rewards}
    
    elif cfg_training.name == "sft":
        pol_chosen_outputs = model(**chosen_conv_tokens)
        if cfg_method.name == "adapter":
            pass
            # rec_loss_chosen = model.module.perturbation_layer._current_recon_loss.clone()
            # chose_gate_value = model.module.perturbation_layer._current_gate_value.clone()
        pol_chosen_logprobs = compute_logprobs(pol_chosen_outputs.logits, chosen_labels, selection_mask=chosen_mask)

        sft_loss = sft(pol_chosen_logprobs)

        if not cfg_method.name == "adapter":
            return {"sft_loss": sft_loss}
        else:
            # sft_loss = max(step_ratio, 0.2)*step_ratio*sft_loss + min(0.8, 1-step_ratio)*rec_loss_chosen
            # wd = chose_gate_value.mean()**2
            # sft_loss = sft_loss + 0.1 * wd
            return {"sft_loss": sft_loss}
    
    elif cfg_training.name == "ga":
        pol_rejected_outputs = model(**rejected_conv_tokens)
        if cfg_method.name == "adapter":
            pass
            # rec_loss_rejected = model.module.perturbation_layer._current_recon_loss.clone()
            # rej_gate_value = model.module.perturbation_layer._current_gate_value.clone()
        pol_rejected_logprobs = compute_logprobs(pol_rejected_outputs.logits, rejected_labels, selection_mask=rejected_mask)

        unlearn_loss = unlearn(pol_rejected_logprobs)

        if not cfg_method.name == "adapter":
            return {"ga": unlearn_loss}
        else:
            # unlearn_loss = max(step_ratio, 0.2)*step_ratio*unlearn_loss + rec_loss_rejected*min(0.8, 1-step_ratio)
            # wd = rej_gate_value.mean()**2
            # unlearn_loss = unlearn_loss + 0.1 * wd
            return {"ga": unlearn_loss}
    
    elif cfg_training.name == "npo":
        pol_rejected_outputs = model(**rejected_conv_tokens)

        if cfg_method.name == "adapter":
            pass
            # rec_loss_rejected = model.module.perturbation_layer._current_recon_loss.clone()
            # rej_gate_value = model.module.perturbation_layer._current_gate_value.clone()
        pol_rejected_logprobs = compute_logprobs(pol_rejected_outputs.logits, rejected_labels, selection_mask=rejected_mask)

        unlearn_loss = npo(pol_rejected_logprobs, ref_rejected_logprobs, beta=cfg_training.beta)

        if not cfg_method.name == "adapter":
            return {"npo": unlearn_loss}
        else:
            # unlearn_loss = max(step_ratio, 0.2)*unlearn_loss + rec_loss_rejected*min(0.8, 1-step_ratio)
            # wd = rej_gate_value.mean()**2
            # unlearn_loss = unlearn_loss + 0.1 * wd
            # return {"npo": unlearn_loss,
            #         "reconstruction_loss": rec_loss_rejected}
            return {"npo": unlearn_loss}

    else:
        raise NotImplementedError(f"{cfg_training.name} not supported!")
            

def train_model(model, 
                processor, 
                train_dl, 
                max_epochs, 
                accumulation_steps, 
                early_stopping, 
                optimizer,
                scheduler,
                cfg_training,
                total_steps,
                image_resizer,
                cfg_method,
                save_path,
                logger=None):    
    
    model = model.to(get_device_id())

    steps_counter = 0
    save_checkpoints = [int(perc/100 * total_steps) for perc in cfg_training.save_checkpoints]

    for epoch in range(max_epochs):
        if use_distributed():
            train_dl.sampler.set_epoch(epoch)
        
        model.train()
        for i, batch in enumerate(progress(train_dl, description=f"Epoch {epoch+1}/{max_epochs}. \
                                           Training {Color.magenta(cfg_method.name)} with Loss {Color.magenta(cfg_training.name)}")):
            
            
            
            if ((i+1) % accumulation_steps) == 0:
                # Count only effective steps
                steps_counter += 1

                output = model_step(model=model, 
                                    processor=processor, 
                                    batch=batch, 
                                    transform=image_resizer, 
                                    cfg_training=cfg_training, 
                                    cfg_method=cfg_method, 
                                    step_ratio=(steps_counter/total_steps))

                # Loss to backprop is always the first
                loss_out = next(iter(output.values()))

                if logger is not None:
                    logger.info(f"Step {i} | " + " | ".join([f"{key}: {value.item():.4f}" for key, value in output.items()]))
                    logger.log_metrics({key: value.item() for key, value in output.items()})
                
                (loss_out / accumulation_steps).backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if use_distributed():
                    barrier()

                # This is basically not used anymore, so here save_model and stop_training are false till now
                save_model, stop_training = early_stopping.check(model, processor)

                if steps_counter in save_checkpoints:
                    save_model = True
                    perc = cfg_training.save_checkpoints[save_checkpoints.index(steps_counter)]
                    save_path_step = save_path + f"/perc_{perc}"
                    os.makedirs(save_path_step, exist_ok=True)

                if save_model:
                    if is_main_process():
                        logger.info(f"Saving model to {save_path_step}")
                        
                        if cfg_method.requires_perturbation:
                            if use_distributed():
                                model.module.save_pretrained(save_path_step)
                            else:
                                model.save_pretrained(save_path_step)

                        elif cfg_method.use_lora:
                            if use_distributed():
                                save_lora(model.module, save_path_step)
                            else:
                                save_lora(model, save_path_step) 

                        else:
                            if use_distributed():
                                torch.save(model.module.state_dict(), save_path_step+f"/parameters_trained.pt")
                            else:
                                torch.save(model.state_dict(), save_path_step+f"/parameters_trained.pt")

                if stop_training:
                    logger.info(f"Early stopping at epoch {epoch+1}, step {steps_counter}")
                    return model

            else:
                with model.no_sync() if use_distributed() else nullcontext():
                    output = model_step(model=model, 
                                    processor=processor, 
                                    batch=batch, 
                                    transform=image_resizer, 
                                    cfg_training=cfg_training, 
                                    cfg_method=cfg_method,
                                    step_ratio=(steps_counter/total_steps))
                    
                    loss_out = next(iter(output.values()))
                    (loss_out / accumulation_steps).backward()

    return model

        

    




    

    

    

    

    
    