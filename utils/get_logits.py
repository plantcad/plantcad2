import pandas as pd
import torch, sys, h5py, os
import numpy as np
import argparse, os
from tqdm import tqdm
import logging

def maskedTokenLogit(model, tokenizer, loader, device, output_path):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        with h5py.File(output_path, 'r+') as hf:
            if 'predicted_logits' not in hf or 'true_token_ids' not in hf or 'processed_batches' not in hf.attrs:
                raise ValueError("Incomplete h5py file")
    except (OSError, ValueError):
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Deleted incomplete file: {output_path}")
        logger.info(f"Creating new file: {output_path}")

    with h5py.File(output_path, 'a') as hf:
        # Initialize processed_batches
        if 'processed_batches' not in hf.attrs:
            hf.attrs['processed_batches'] = 0
        processed_batches = hf.attrs['processed_batches']
        feature_dim = 8

        if 'predicted_logits' not in hf or 'true_token_ids' not in hf:
            if 'predicted_logits' not in hf:
                hf.create_dataset('predicted_logits',
                                  shape=(0, feature_dim),
                                  maxshape=(None, feature_dim),
                                  chunks=True,
                                  compression="gzip")
            if 'true_token_ids' not in hf:
                hf.create_dataset('true_token_ids',
                                  shape=(0,),
                                  maxshape=(None,),
                                  chunks=True,
                                  compression="gzip")

        logits_dataset = hf['predicted_logits']
        true_ids_dataset = hf['true_token_ids']

        # Skip already processed batches
        for batch_idx, batch in enumerate(tqdm(loader, desc="Getting masked token logits"), start=0):
            if batch_idx < processed_batches:
                continue

            curIDs = batch['masked_ids'].to(device).squeeze(1)
            true_token_ids = batch['true_ids'].to(device)  
            true_token_ids = true_token_ids.flatten()

            with torch.inference_mode():
                all_logits = model(input_ids=curIDs).logits

            # Select logits for all masked positions
            masked_positions = (curIDs == tokenizer.mask_token_id).unsqueeze(-1).expand(-1, -1, all_logits.size(-1))
            predicted_logits = torch.masked_select(all_logits, masked_positions).view(-1, all_logits.size(-1))


            # Resize and append datasets to disk
            current_size_logits = logits_dataset.shape[0]
            new_size_logits = current_size_logits + predicted_logits.shape[0]
            logits_dataset.resize((new_size_logits, feature_dim))
            logits_dataset[current_size_logits:new_size_logits, :] = predicted_logits.cpu().numpy()

            current_size_ids = true_ids_dataset.shape[0]
            new_size_ids = current_size_ids + true_token_ids.size(0)
            true_ids_dataset.resize((new_size_ids,))
            true_ids_dataset[current_size_ids:new_size_ids] = true_token_ids.cpu().numpy()

            # Update processed_batches and write to disk
            hf.attrs['processed_batches'] = batch_idx + 1
            hf.flush()

    logger.info(f"Finished processing. Total batches processed: {batch_idx + 1}")
    return True