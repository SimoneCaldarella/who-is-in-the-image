import torch
import os
from tutus.modeling import generate_predictions
from tutus.utils.dist import save_files_from_ranks, aggregate_files_from_ranks, use_distributed
from tutus.utils import progress
import pandas as pd
from tutus.utils import Color
from tutus.utils import Logger


def generate_pristine(model, processor, dataloader, file_path):
    if os.path.exists(file_path):
        Logger.info(f"File {file_path} already exists. Skipping generation.")
        return 200
    
    else:
        outfolder, filename = file_path.rsplit("/", 1)
        os.makedirs(outfolder, exist_ok=True)
        model.eval()
        columns = ["name", "image_path", "support", "leaking"]

        sample = {k: [] for k in columns}
        predictions_df = pd.DataFrame(sample)

        for i, batch in enumerate(progress(dataloader, description=f"Processing dataset with {Color.magenta('PRISTINE')}")):

            if i%100 == 0:
                Logger.info(f"Processed {i} samples")

            names, image_paths, leaking_prompts, support_prompts = batch
            leaking_prompts = [{"question": l_prompt} for l_prompt in leaking_prompts]
            support_prompts = [{"question": s_prompt} for s_prompt in support_prompts]

            support_outputs = generate_predictions(model, processor, support_prompts, image_paths, use_distributed=use_distributed())
            leaking_outputs = generate_predictions(model, processor, leaking_prompts, image_paths, use_distributed=use_distributed())


            new_predictions = pd.DataFrame({"name": names,
                                            "image_path": image_paths,
                                            "support": support_outputs,
                                            "leaking": leaking_outputs})

            predictions_df = pd.concat([predictions_df, new_predictions], ignore_index=True)

        # Save predictions from each rank to a file in a temp folder
        rank_filename = save_files_from_ranks(predictions_df, outfolder, filename, "csv")
        Logger.info("Saving predictions to {}".format(rank_filename))

        if use_distributed():
            # Aggregate all predictions
            aggregate_files_from_ranks(outfolder, filename)
            Logger.info("Aggregated predictions to {}".format(file_path))

        return 200