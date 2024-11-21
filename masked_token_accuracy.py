import h5py
import sys
import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from utils.data_utils import maskedTokenDataset, load_fastas
from utils.get_logits import maskedTokenLogit
from utils.model_utils import load_model


def parse_arguments():
    """Parses command-line arguments."""
    parser = ArgumentParser(description="Masked token accuracy")
    parser.add_argument("-input", dest="inputFASTA", type=str, required=True,
                        help="The directory of input fasta")
    parser.add_argument("-output", dest="output", type=str, required=True,
                        help="The directory of output")
    parser.add_argument("-model", dest="modelDir", type=str, required=True,
                        help="The directory of pre-trained model")
    parser.add_argument("-device", dest="device", type=str, default="cuda:0",
                        help="The device to run the model")
    parser.add_argument("-batchSize", dest="batchSize", type=int, default=128,
                        help="The batch size for the model")
    parser.add_argument("-tokenIdx", dest="tokenIdx", type=int, default=255,
                        help="The index of the token to be masked")
    return parser.parse_args()


def calculate_accuracy_and_loss(logits, true_ids, loss_fct, device):
    """
    Calculates the loss, perplexity, and accuracy for the masked token predictions.
    """
    logits = logits.to(device)
    true_ids = true_ids.long().to(device)
    loss = loss_fct(logits, true_ids)

    predictions = torch.argmax(logits, dim=1)
    correct_predictions = torch.sum(predictions == true_ids).item()
    total_predictions = len(true_ids)

    avg_loss = loss.item()
    perplexity = torch.exp(torch.tensor(avg_loss))
    accuracy = correct_predictions / total_predictions

    return avg_loss, perplexity.item(), accuracy


def main():
    # Parse arguments
    args = parse_arguments()

    # Load model, tokenizer, and dataset
    model, tokenizer = load_model(args)
    sequences, names = load_fastas(args)
    dataset = maskedTokenDataset(
        sequences=sequences,
        tokenizer=tokenizer,
        tokenIdx=args.tokenIdx
    )
    loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False, num_workers=1)

    # Generate logits and save to HDF5
    maskedTokenLogit(model, tokenizer, loader, args.device, args.output)
    torch.cuda.empty_cache()

    # Load saved logits and calculate metrics
    loss_fct = torch.nn.CrossEntropyLoss()
    with h5py.File(args.output, 'a') as hf:
        logits_dataset = torch.tensor(hf['predicted_logits'][:])
        true_ids_dataset = torch.tensor(hf['true_token_ids'][:])

    avg_loss, perplexity, accuracy = calculate_accuracy_and_loss(
        logits_dataset, true_ids_dataset, loss_fct, args.device
    )

    # Print results
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Masked Token Accuracy: {accuracy:.4%}")


if __name__ == '__main__':
    main()