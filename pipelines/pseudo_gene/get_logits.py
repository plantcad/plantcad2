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


def chunks(lst, n):
    return [''.join(lst[i:i+n]).upper() for i in range(0, len(lst), n)]

def predicted_masked_tokens(model_version, junction_idx, tokenizer, base_dir=None, chunk_size=None, JUNCTIONS=None):
    """
    Process data from a specific model version for a given junction type.
    
    Args:
        model_version: Model version key (e.g., 'pcv1', 'pcv2_1')
        junction_idx: Index of the junction in the JUNCTIONS list
        tokenizer: Tokenizer object for converting IDs to tokens
        base_dir: Optional base directory for file paths
        chunk_size: Optional dictionary to override default chunk sizes
        
    Returns:
        tuple: (predicted_codon, true_tokens)
    """
    from config import JUNCTIONS, get_file_paths, get_file_paths_list, reverse_complement
    
    junction = JUNCTIONS[junction_idx]
    
    file_paths = get_file_paths_list(model_version, base_dir)
    file_path = file_paths[junction_idx]
    
    if chunk_size is None:
        chunk_size = {
            'start_sites': 3, 
            'stop_sites': 3,
            'donor': 2,
            'acceptor': 2
        }
    
    with h5py.File(file_path, 'a') as hf:
        model_logits = torch.tensor(hf['predicted_logits'][:])
        true_ids = torch.tensor(hf['true_token_ids'][:])
    
    predicted_tokens = tokenizer.convert_ids_to_tokens(torch.argmax(model_logits, dim=1))
    
    n_tokens = chunk_size[junction]
    
    predicted_codon = chunks(predicted_tokens, n=n_tokens)
    true_tokens = chunks(tokenizer.convert_ids_to_tokens(true_ids), n=n_tokens)
    
    return predicted_codon, true_tokens