import argparse
import logging
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from captum.attr import IntegratedGradients

# Configure logging to provide informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(model_name: str, peft_model_path: str, task_type: str = "classification"):
    """
    Loads the base model, tokenizer, and applies the PEFT adapter.

    This function also wraps the base model's forward pass to gracefully handle
    extra arguments passed by the PEFT wrapper.

    Args:
        model_name: The name or path of the base transformer model.
        peft_model_path: The path to the PEFT model checkpoint.
        task_type: The type of task, either "classification" or "regression".

    Returns:
        A tuple containing the fully loaded PEFT model and its corresponding tokenizer.
    """
    logging.info(f"Loading base model: {model_name}")
    if task_type == "classification":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
        )
    else:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=1,
            problem_type="regression",
        )

    # Wrap the original forward method to ignore unused arguments
    original_forward = base_model.forward
    def forgiving_forward(*args, **kwargs):
        for key in ['attention_mask', 'output_attentions', 'output_hidden_states']:
            kwargs.pop(key, None)
        return original_forward(*args, **kwargs)
    base_model.forward = forgiving_forward

    logging.info(f"Loading PEFT model from: {peft_model_path}")
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

class CaduceusCaptumWrapper(nn.Module):
    """
    A wrapper for the Caduceus model to ensure its compatibility with Captum.

    Captum's attribution methods require a model that accepts `inputs_embeds`
    and returns logits. This wrapper provides that interface.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds):
        """
        Defines the forward pass for the wrapper.
        """
        outputs = self.model(input_ids=None, inputs_embeds=inputs_embeds)
        return outputs.logits

def create_rcps_embeddings(input_ids: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Creates Reverse-Complement-and-Stack (RCPS) embeddings for Caduceus.

    Args:
        input_ids: A tensor of token IDs.
        model: The model containing the necessary embedding layers.

    Returns:
        A tensor of the final RCPS embeddings.
    """
    word_embedding_module = model.base_model.caduceus.backbone.embeddings.word_embeddings
    embedding_layer = word_embedding_module.embedding
    rc_layer = word_embedding_module.rc
    
    # Create forward and reverse-complement embeddings
    fwd_out = embedding_layer(input_ids)
    rc_ids = rc_layer(input_ids)
    rc_out = torch.flip(embedding_layer(rc_ids), dims=[-2, -1])
    
    # Concatenate to create the final stacked embeddings
    return torch.cat([fwd_out, rc_out], dim=-1)

def calculate_attributions(
    dna_sequence: str,
    tokenizer: AutoTokenizer,
    model: nn.Module,
    ig: IntegratedGradients,
    device: torch.device,
    target_class_index: int = 1,
) -> np.ndarray:
    """
    Calculates Integrated Gradients attributions for a single DNA sequence.

    Args:
        dna_sequence: The input DNA sequence string.
        tokenizer: The tokenizer for the model.
        model: The model to be analyzed.
        ig: An instance of the IntegratedGradients class.
        device: The device (CPU or CUDA) to run computations on.
        target_class_index: The target class index for attribution calculation.

    Returns:
        A NumPy array containing the summarized attribution scores.
    """
    # Tokenize the sequence and create a shuffled baseline for attribution
    inputs = tokenizer(dna_sequence, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    
    shuffled_dna_list = list(dna_sequence)
    np.random.shuffle(shuffled_dna_list)
    shuffled_dna = "".join(shuffled_dna_list)
    baseline_inputs = tokenizer(shuffled_dna, return_tensors="pt")
    baseline_ids = baseline_inputs['input_ids'].to(device)

    # Generate RCPS embeddings for both the input and the baseline
    inputs_embeds = create_rcps_embeddings(input_ids, model)
    baseline_embeds = create_rcps_embeddings(baseline_ids, model)

    # Compute attributions using Integrated Gradients
    attributions = ig.attribute(
        inputs_embeds,
        baselines=baseline_embeds,
        target=target_class_index,
        internal_batch_size=5,
        n_steps=50
    )
    
    # Summarize and return attributions
    return attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()

def load_sequences_from_tsv(file_path: str, sequence_length: int) -> list:
    """
    Loads and validates DNA sequences from a specified TSV file.

    Args:
        file_path: The path to the TSV file.
        sequence_length: The expected length of the DNA sequences.

    Returns:
        A list of valid DNA sequences.
    """
    logging.info(f"Loading sequences from {file_path}...")
    try:
        df = pd.read_csv(file_path, sep='\t')
        if 'seq' not in df.columns:
            raise ValueError("The TSV file must contain a 'Seq' column.")
        
        sequences = df['seq'].dropna().astype(str).tolist()
        
        # Filter sequences to ensure they match the required length
        original_count = len(sequences)
        sequences = [s for s in sequences if len(s) == sequence_length]
        if len(sequences) < original_count:
            logging.warning(
                f"Filtered out {original_count - len(sequences)} sequences "
                f"that were not {sequence_length} characters long."
            )
            
        logging.info(f"Successfully loaded {len(sequences)} sequences.")
        return sequences
        
    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")
        sys.exit(1)

def main():
    """
    Main function to orchestrate the attribution analysis pipeline.
    """
    parser = argparse.ArgumentParser(description="Calculate attributions for DNA sequences using a fine-tuned Caduceus model.")
    parser.add_argument("--checkpoint", help="Path to the fine-tuned PEFT model checkpoint.")
    parser.add_argument("--input", help="Path to the TSV file containing DNA sequences in a 'Seq' column.")
    parser.add_argument("--model_name", help="The base model name or path from Hugging Face.")
    parser.add_argument("--sequence_length", type=int, default=2048, help="The expected length of DNA sequences.")
    parser.add_argument("--output", default="attribution_matrix.npz", help="The output file path for the compressed attribution matrix.")
    parser.add_argument("--device", default="cuda:0", help="The device used to get attributions.")
    args = parser.parse_args()

    device = args.device
    # Set up the device, model, and tokenizer
    logging.info(f"Using device: {device}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.checkpoint)
    model.to(device)
    model.eval()

    # Initialize Captum with the wrapped model
    wrapped_model = CaduceusCaptumWrapper(model)
    ig = IntegratedGradients(wrapped_model)

    # Load DNA sequences from the specified file
    sequences = load_sequences_from_tsv(args.input, args.sequence_length)
    if not sequences:
        logging.info("No valid sequences were loaded. Exiting.")
        return

    # Process each sequence to calculate attributions
    all_attributions = []
    for seq in tqdm(sequences, desc="Analyzing Sequences"):
        try:
            attr = calculate_attributions(seq, tokenizer, model, ig, device)
            all_attributions.append(attr)
        except Exception as e:
            logging.error(f"Skipping a sequence due to an error: {e}", exc_info=True)

    # Save the aggregated results
    if all_attributions:
        attribution_matrix = np.array(all_attributions)
        np.savez_compressed(args.output, attr=attribution_matrix)
        logging.info(f"Saved attribution matrix with shape {attribution_matrix.shape} to {args.output}")
    else:
        logging.warning("No attributions were successfully calculated. No output file was saved.")

if __name__ == "__main__":
    main()
