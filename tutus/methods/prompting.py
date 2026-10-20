import os
import pandas as pd

from tutus.modeling import generate_predictions
from tutus.utils.dist import save_files_from_ranks, aggregate_files_from_ranks, use_distributed
from tutus.utils import progress
from tutus.utils import Color, Logger, progress

def generate_prompting(model, processor, dataloader, file_path, system_prompt, jailbreak=None):

    outfolder, filename = file_path.rsplit("/", 1)
    os.makedirs(outfolder, exist_ok=True)
    model.eval()
    columns = ["name", "image_path", "support", "leaking"]

    sample = {k: [] for k in columns}
    predictions_df = pd.DataFrame(sample)

    for i, batch in enumerate(progress(dataloader, description=f"Processing dataset with {Color.magenta('PROMPTING')}")):

        names, image_paths, leaking_prompts, support_prompts = batch

        # Adding instruction not to reveal identity
        jailbreak_prompts = [{"system_prompt": system_prompt, "question": l_prompt+jailbreak} for l_prompt in leaking_prompts]
        leaking_prompts = [{"system_prompt": system_prompt, "question": l_prompt} for l_prompt in leaking_prompts]
        support_prompts = [{"system_prompt": system_prompt, "question": s_prompt} for s_prompt in support_prompts]
        


        support_outputs = generate_predictions(model, processor, support_prompts, image_paths, use_distributed=use_distributed())
        leaking_outputs = generate_predictions(model, processor, leaking_prompts, image_paths, use_distributed=use_distributed())
        jailbreak_outputs = generate_predictions(model, processor, jailbreak_prompts, image_paths, use_distributed=use_distributed())


        new_predictions = pd.DataFrame({"name": names,
                                        "image_path": image_paths,
                                        "support": support_outputs,
                                        "leaking": leaking_outputs,
                                        "jailbreak": jailbreak_outputs})

        predictions_df = pd.concat([predictions_df, new_predictions], ignore_index=True)

    # Save predictions from each rank to a file in a temp folder
    rank_filename = save_files_from_ranks(predictions_df, outfolder, filename, "csv")
    Logger.info("Saving predictions to {}".format(rank_filename))

    if use_distributed():
        # Aggregate all predictions
        aggregate_files_from_ranks(outfolder, filename)
        Logger.info("Aggregated predictions to {}".format(file_path))

    return 200