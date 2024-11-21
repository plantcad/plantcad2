import torch
import logging
from transformers import AutoModelForMaskedLM, AutoTokenizer

def get_optimal_dtype():
    if not torch.cuda.is_available():
        logging.info("Using float32 as no GPU is available.")
        return torch.float32  

    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)

    if capability[0] >= 8:  # sm_80 or higher
        logging.info("Using bfloat16 as the GPU supports sm_80 or higher.")
        return torch.bfloat16
    elif capability[0] >= 6:  # sm_60 or higher
        logging.info("Using float16 as the GPU supports sm_60 or higher.")
        return torch.float16
    else:
        logging.info("Using float32 as the GPU does not support float16 or bfloat16.")
        return torch.float32


def load_model(args):
    device = args.device
    dtype = get_optimal_dtype()
    model = AutoModelForMaskedLM.from_pretrained(
        args.modelDir, 
        trust_remote_code=True, 
        dtype=dtype
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.modelDir, trust_remote_code=True)
    model.eval()
    return model, tokenizer
