import h5py
import torch

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