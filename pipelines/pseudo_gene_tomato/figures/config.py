import os

JUNCTIONS = ["start_sites", "stop_sites", "donor", "acceptor"]
TAXA = "SL3.0_ITAG3.2_"

PCV1_PATHS = {
    "start_sites": f"{TAXA}start_sites_mask_ATG_PlantCaduceus_l32.h5",
    "stop_sites": f"{TAXA}stop_sites_mask_TAG_PlantCaduceus_l32.h5", 
    "donor": f"{TAXA}donor_mask_GT_PlantCaduceus_l32.h5",
    "acceptor": f"{TAXA}acceptor_mask_AG_PlantCaduceus_l32.h5"
}

PCV2_1_PATHS = {
    "start_sites": f"{TAXA}start_sites_mask_ATG_pcv2-l24-d0768.h5",
    "stop_sites": f"{TAXA}stop_sites_mask_TAG_pcv2-l24-d0768.h5", 
    "donor": f"{TAXA}donor_mask_GT_pcv2-l24-d0768.h5",
    "acceptor": f"{TAXA}acceptor_mask_AG_pcv2-l24-d0768.h5"
}

PCV2_2_PATHS = {
    "start_sites": f"{TAXA}start_sites_mask_ATG_pcv2-l48-d1024.h5",
    "stop_sites": f"{TAXA}stop_sites_mask_TAG_pcv2-l48-d1024.h5", 
    "donor": f"{TAXA}donor_mask_GT_pcv2-l48-d1024.h5",
    "acceptor": f"{TAXA}acceptor_mask_AG_pcv2-l48-d1024.h5"
}

PCV2_3_PATHS = {
    "start_sites": f"{TAXA}start_sites_mask_ATG_pcv2-l48-d1536.h5",
    "stop_sites": f"{TAXA}stop_sites_mask_TAG_pcv2-l48-d1536.h5", 
    "donor": f"{TAXA}donor_mask_GT_pcv2-l48-d1536.h5",
    "acceptor": f"{TAXA}acceptor_mask_AG_pcv2-l48-d1536.h5"
}

EVO2_FWD_PATHS = {
    "start_sites": f"{TAXA}start_sites_evo2_7b_ntokens_3.tsv",
    "stop_sites": f"{TAXA}stop_sites_evo2_7b_ntokens_3.tsv",
    "donor": f"{TAXA}donor_filtered_evo2_7b_ntokens_2.tsv",
    "acceptor": f"{TAXA}acceptor_filtered_evo2_7b_ntokens_2.tsv"
}

EVO2_RC_PATHS = {
    "start_sites": f"{TAXA}start_sites_evo2_7b_rc_ntokens_3.tsv",
    "stop_sites": f"{TAXA}stop_sites_evo2_7b_rc_ntokens_3.tsv",
    "donor": f"{TAXA}donor_evo2_7b_rc_ntokens_2.tsv",
    "acceptor": f"{TAXA}acceptor_evo2_7b_rc_ntokens_2.tsv"
}

LABEL_PATHS = {
    "start_sites": f"{TAXA}start_filtered_labels_rmless8192.tsv",
    "stop_sites": f"{TAXA}stop_filtered_labels_rmless8192.tsv",
    "donor": f"{TAXA}donor_filtered_labels_rmless8192.tsv",
    "acceptor": f"{TAXA}acceptor_filtered_labels_rmless8192.tsv"
}

# Group all path dictionaries for easy access
ALL_PATHS = {
    "pcv1": PCV1_PATHS,
    "pcv2_1": PCV2_1_PATHS,
    "pcv2_2": PCV2_2_PATHS,
    "pcv2_3": PCV2_3_PATHS,
    "evo2_fwd": EVO2_FWD_PATHS,
    "evo2_rc": EVO2_RC_PATHS,
    "labels": LABEL_PATHS
}

def get_file_paths(model_version, base_dir=None):
    """
    Get file paths for a specific model version.
    
    Args:
        model_version: Model version key (e.g., 'pcv1', 'evo2_rc')
        base_dir: Optional base directory
    
    Returns:
        Dictionary mapping junction types to their full file paths
    """
    if model_version not in ALL_PATHS:
        valid_versions = list(ALL_PATHS.keys())
        raise KeyError(f"Invalid model version: {model_version}. Valid options: {valid_versions}")
    
    base = base_dir or ""
    return {junction: os.path.join(base, ALL_PATHS[model_version][junction]) 
            for junction in JUNCTIONS}

def get_file_paths_list(model_version, base_dir=None):
    """
    Get a list of file paths for all junction types for the specified model version.
    
    Args:
        model_version: Model version key
        base_dir: Optional base directory
    
    Returns:
        List of file paths
    """
    paths_dict = get_file_paths(model_version, base_dir)
    return [paths_dict[junction] for junction in JUNCTIONS]

def reverse_complement(seq):
    """
    Return the reverse complement of a DNA sequence.
    
    Args:
        seq: DNA sequence string
        
    Returns:
        Reverse complemented sequence
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
                  'a': 't', 'c': 'g', 'g': 'c', 't': 'a',
                  'N': 'N', 'n': 'n'}
    
    return ''.join(complement.get(base, base) for base in reversed(seq))