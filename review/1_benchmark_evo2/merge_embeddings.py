import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_embeddings(output_pkl, delete_parts=False):
    output_path = Path(output_pkl)
    
    # Check if the output path has a parent directory
    parent_dir = output_path.parent
    # If the parent is empty (current dir), make it absolute or use cwd
    if str(parent_dir) == '.':
        parent_dir = Path.cwd()
        
    base_name = output_path.stem # 'embeddings' (if file is embeddings.pkl)
    
    # Construct search pattern
    # The script looks for .rank*.pkl variants.
    search_pattern = f"{base_name}.rank*.pkl"
    files = sorted(list(parent_dir.glob(search_pattern)))
    
    if not files:
        logging.error(f"No partial files found matching pattern '{search_pattern}' in {parent_dir}")
        return

    logging.info(f"Found {len(files)} partial files.")
    
    # Parse ranks to ensure correct ordering
    rank_files = {}
    for f in files:
        try:
            # Expected format: name.rankN.pkl
            # Split by '.rank'
            parts = f.name.split('.rank')
            if len(parts) < 2:
                continue
            
            # The part after .rank should be "N.pkl"
            rank_part = parts[-1]
            if not rank_part.endswith('.pkl'):
                continue
                
            rank_str = rank_part[:-4] # remove .pkl
            rank = int(rank_str)
            rank_files[rank] = f
        except ValueError:
            logging.warning(f"Skipping file with unexpected format: {f.name}")
            continue

    sorted_ranks = sorted(rank_files.keys())
    if not sorted_ranks:
        logging.error("No valid rank files found.")
        return

    # specific check for 0-indexed contiguous ranks
    expected_ranks = list(range(len(sorted_ranks)))
    if sorted_ranks != expected_ranks:
        logging.warning("Ranks are not contiguous starting from 0.")
        logging.warning(f"Found ranks: {sorted_ranks}")
        logging.info("Proceeding with sorted ranks...")
    
    # Pass 1: Calculate total size and determine schema
    logging.info("Pass 1: Identifying schema and calculating total dataset size...")
    total_samples = 0
    label_dtype = None
    embedding_keys = [] # list of keys that hold embeddings (e.g. forward, reverse, mean, max)
    embedding_dims = {} # key -> dim
    
    # Peek at first file for metadata and schema detection
    first_file = rank_files[sorted_ranks[0]]
    with open(first_file, 'rb') as f:
        data = pickle.load(f)
        
        # Check for labels
        if 'labels' in data:
            label_dtype = data['labels'].dtype
        else:
            logging.error("First file missing 'labels' key.")
            return

        # Identify embedding keys dynamically
        # We look for specific known keys from evo2 and plantcad2
        potential_keys = ['forward', 'reverse', 'mean', 'max']
        
        for k in potential_keys:
            if k in data:
                val = data[k]
                if val is not None and isinstance(val, np.ndarray):
                    embedding_keys.append(k)
                    embedding_dims[k] = val.shape[1]
                    logging.info(f"  Found embedding key: '{k}' with dim {val.shape[1]}")
                elif val is None:
                    logging.info(f"  Found key: '{k}' but it is None (skipping)")
        
        if not embedding_keys:
            logging.error("No valid embedding arrays (forward, reverse, mean, max) found in first file.")
            return
            
        del data
        gc.collect()
    
    # Sum up total samples
    for r in tqdm(sorted_ranks, desc="Scanning files"):
        fpath = rank_files[r]
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            n = len(data['labels'])
            total_samples += n
            
            # Sanity checks
            for k in embedding_keys:
                if k not in data:
                     raise ValueError(f"Key '{k}' missing in {fpath.name}")
                if data[k] is None:
                    continue # Should we error? Assumes consistency.
                if data[k].shape[1] != embedding_dims[k]:
                    raise ValueError(f"Dimension mismatch for '{k}' in {fpath.name}")
            
            del data
            gc.collect()
            
    logging.info(f"Total samples to merge: {total_samples}")
    
    # Estimates
    bytes_per_float = 4
    total_bytes = 0
    for k in embedding_keys:
        total_bytes += total_samples * embedding_dims[k] * bytes_per_float
    # Add labels (assuming int64 usually, or whatever dtype)
    total_bytes += total_samples * 8 
    
    logging.info(f"Estimated memory required for arrays: {total_bytes / (1024**3):.2f} GB")
    
    # Allocate arrays
    logging.info("Allocating memory for merged arrays...")
    final_arrays = {}
    try:
        final_labels = np.empty((total_samples,), dtype=label_dtype)
        for k in embedding_keys:
            final_arrays[k] = np.empty((total_samples, embedding_dims[k]), dtype=np.float32)
        all_metadata = []
    except MemoryError:
        logging.error("CRITICAL ERROR: Not enough memory to allocate the final merged arrays.")
        logging.error("Try running this script on a node with more RAM.")
        return

    # Pass 2: Fill arrays
    logging.info("Pass 2: Merging data...")
    current_idx = 0
    
    for r in tqdm(sorted_ranks, desc="Merging"):
        fpath = rank_files[r]
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            
        n = len(data['labels'])
        end_idx = current_idx + n
        
        # Copy data
        final_labels[current_idx:end_idx] = data['labels']
        for k in embedding_keys:
            final_arrays[k][current_idx:end_idx] = data[k]
            
        all_metadata.append(data['metadata'])
        
        current_idx = end_idx
        
        # Free memory immediately
        del data
        gc.collect()

    logging.info("Concatenating metadata...")
    final_metadata = pd.concat(all_metadata, ignore_index=True)
    
    final_data = {
        'labels': final_labels,
        'metadata': final_metadata
    }
    # Add embeddings to final dict
    for k in embedding_keys:
        final_data[k] = final_arrays[k]
        
    # Add any explicit None for missing optional keys if we want to preserve structure exactly?
    # For evo2, 'reverse' might be None. To be safe, if we found 'forward' but not 'reverse', maybe we should add reverse: None?
    # Current logic only adds keys that were ndarrays in the first file.
    # If the user code expects 'reverse' to exist (even if None), we might break it.
    # Check evo2_embedding.py again:
    # `if data['reverse'] is None:` lines 294, 298, 302
    # So the key must exist!
    
    # Restore missing keys as None if they are expected schemas
    if 'forward' in embedding_keys and 'reverse' not in embedding_keys:
        final_data['reverse'] = None
        
    # For PlantCAD2, it has 'mean' and 'max', usually both present.
        
    logging.info(f"Saving merged file to {output_path}...")
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("Successfully saved merged embeddings.")
    except Exception as e:
        logging.error(f"Error saving pickle file: {e}")
        return

    if delete_parts:
        logging.info("Deleting partial files...")
        for r in sorted_ranks:
            os.remove(rank_files[r])
        logging.info("Partial files deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-efficient merge of embeddings (Evo2 or PlantCAD2)")
    parser.add_argument('output_file', help="Path to the final output file. The script looks for .rank*.pkl variants.")
    parser.add_argument('--delete-parts', action='store_true', help="Delete partial rank files after successful merge")
    
    args = parser.parse_args()
    
    merge_embeddings(args.output_file, args.delete_parts)
