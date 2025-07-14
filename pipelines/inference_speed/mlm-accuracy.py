import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
from datasets import load_from_disk
import sys
import os


class DNADataset(Dataset):
    """Simple DNA sequence dataset for MLM evaluation"""
    
    def __init__(self, sequences: List[str], tokenizer, max_length: int = 8192):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
        }


def evaluate_mlm_accuracy(model_path: str, 
                         sequences: List[str] = None,
                         seq_length: int = 512,
                         batch_size: int = 16,
                         mlm_probability: float = 0.15,
                         device: str = "cuda") -> Dict:
    """
    Evaluate masked language modeling accuracy using HF default recipe
    
    Args:
        model_path: HuggingFace model path
        sequences: List of DNA sequences (if None, generates random ones)
        seq_length: Length of each sequence
        batch_size: Batch size for evaluation
        mlm_probability: Probability of masking tokens (HF default: 0.15)
        device: Device to run on
    
    Returns:
        Dictionary with accuracy metrics
    """
    
    print(f"Loading model: {model_path}")
    
    # Load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    
    print(f"Using {len(sequences)} provided sequences")
    
    # Create dataset
    dataset = DNADataset(sequences, tokenizer, seq_length)
    
    # Data collator for MLM (HF default recipe)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        return_tensors="pt"
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    print(f"Evaluating MLM accuracy with {mlm_probability:.1%} masking probability...")
    
    total_predictions = 0
    correct_predictions = 0
    total_masked_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating MLM")):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            
            # Get predictions (assuming model outputs logits)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 'last_hidden_state'):
                # If no head, assume we need to add prediction head or use last layer
                logits = outputs.last_hidden_state
                print("Warning: Model may not have MLM head, using last hidden state")
            else:
                raise ValueError("Cannot find model logits or hidden states")
            
            # Find masked positions
            masked_positions = (labels != -100)
            
            if masked_positions.sum() == 0:
                continue
                
            # Get predictions for masked positions
            masked_logits = logits[masked_positions]
            masked_labels = labels[masked_positions]
            
            # Get predicted tokens
            predicted_tokens = torch.argmax(masked_logits, dim=-1)
            
            # Calculate accuracy
            correct = (predicted_tokens == masked_labels).sum().item()
            total = masked_labels.size(0)
            
            correct_predictions += correct
            total_predictions += total
            total_masked_tokens += masked_positions.sum().item()
            
    
    # Calculate final metrics
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    results = {
        'accuracy': accuracy,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'total_masked_tokens': total_masked_tokens,
        'mlm_probability': mlm_probability,
        'n_sequences': len(sequences),
        'seq_length': seq_length
    }
    
    print(f"\nMLM Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"  Total masked tokens: {total_masked_tokens:,}")
    print(f"  Correct predictions: {correct_predictions:,}")
    print(f"  Total predictions: {total_predictions:,}")
    
    return results


# Load dataset
raw_datasets = load_from_disk('/workdir/jz963/datasets/2024_Phytozome_Dec_Update/Phytozome_8192')
df = raw_datasets['test'].to_pandas() # to pandas dataframe

assemblies = df['assembly'].unique().tolist()

model_path = sys.argv[1]
model_name = os.path.basename(model_path)

# Create output file
output_file = f"mlm_results_{model_name}.txt"

# Write header
with open(output_file, 'w') as f:
    f.write("assembly\taccuracy\ttotal_masked_tokens\tcorrect_predictions\ttotal_predictions\n")

for assembly in assemblies:
    print(f"Evaluating MLM accuracy for assembly: {assembly}")
    curDF = df[df['assembly'] == assembly]
    seq = curDF['seq'].tolist()
    
    result = evaluate_mlm_accuracy(model_path=model_path, 
                            sequences=seq,
                            seq_length=8192,
                            batch_size=32,
                            mlm_probability=0.15,
                            device="cuda:0")
    
    # Append result to file
    with open(output_file, 'a') as f:
        f.write(f"{assembly}\t{result['accuracy']:.4f}\t{result['total_masked_tokens']}\t{result['correct_predictions']}\t{result['total_predictions']}\n")

print(f"\nResults saved to: {output_file}")