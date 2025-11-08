from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embeddings."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLPClassifier(nn.Module):
    """
    Simple MLP classifier for embeddings.

    num_hidden_layers=1: embedding_dim -> 1 (just one linear layer)
    num_hidden_layers=2: embedding_dim -> hidden_dim -> 1 (two linear layers)
    num_hidden_layers=3: embedding_dim -> hidden_dim -> hidden_dim -> 1 (three linear layers)
    """
    def __init__(self, embedding_dim, hidden_dim=256, num_hidden_layers=1, dropout=0.1):
        super().__init__()

        layers = []

        if num_hidden_layers == 1:
            # Just a single linear layer: embedding_dim -> 1
            layers.append(nn.Linear(embedding_dim, 1))
        else:
            # First layer: embedding_dim -> hidden_dim
            layers.append(nn.Linear(embedding_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            # Middle layers: hidden_dim -> hidden_dim
            for _ in range(num_hidden_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            # Output layer: hidden_dim -> 1
            layers.append(nn.Linear(hidden_dim, 1))

        self.fc_layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, embedding_dim)
        returns: (batch,) probabilities
        """
        logits = self.fc_layers(x)  # (batch, 1)
        probs = self.sigmoid(logits).squeeze(-1)  # (batch,)
        return probs


def load_model_and_tokenizer(model_dir, device):
    """Load model and tokenizer with optimal dtype for the GPU."""
    logging.info(f"Loading model and tokenizer from {model_dir}")

    # Determine the appropriate dtype based on the GPU capabilities
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

    optimal_dtype = get_optimal_dtype()

    # Load the model with the selected dtype
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=optimal_dtype)
    except Exception as e:
        logging.error(f"Failed to load model with {optimal_dtype}, falling back to float32. Error: {e}")
        model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer


class SequenceDataset(Dataset):
    """Dataset for sequences."""
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch, tokenizer):
    """Collate function for batching sequences. PlantCAD2 uses Mamba (no attention mask)."""
    # Tokenize the batch - tokenizer() handles batches properly
    # No padding needed if all sequences are the same length
    encodings = tokenizer(
        batch,
        padding=False,
        truncation=False,
        return_tensors='pt',
        return_attention_mask=False,
        return_token_type_ids=False
    )
    # Return only input_ids (Mamba doesn't use attention mask)
    return {'input_ids': encodings['input_ids']}


def extract_embeddings(model, dataloader, device, layer_index=-1):
    """
    Extract embeddings from PlantCAD2 Caduceus model using both mean and max pooling.

    Note: PlantCAD2 uses Caduceus architecture which outputs forward and reverse
    embeddings concatenated. This function splits them, reverses the reverse embedding,
    averages them, then applies both pooling strategies.

    Args:
        model: PlantCAD2 Caduceus model
        dataloader: DataLoader with sequences
        device: Device to use
        layer_index: Which hidden layer to use (-1 for last layer)

    Returns:
        dict with keys 'mean' and 'max', each containing numpy array of shape (num_samples, hidden_dim)
    """
    logging.info(f"Extracting embeddings with both mean and max pooling from PlantCAD2 Caduceus model")
    model.eval()
    all_mean_embeddings = []
    all_max_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            input_ids = batch['input_ids'].to(device)

            with torch.inference_mode():
                outputs = model(input_ids=input_ids, output_hidden_states=True)

            # Get hidden states from specified layer
            token_embeddings = outputs.hidden_states[layer_index]  # (batch, seq_len, hidden_dim * 2)

            # Split forward and reverse embeddings (Caduceus architecture)
            hidden_size = token_embeddings.shape[-1] // 2
            forward = token_embeddings[..., 0:hidden_size]  # (batch, seq_len, hidden_dim)
            reverse = token_embeddings[..., hidden_size:]   # (batch, seq_len, hidden_dim)

            # Reverse the reverse embedding along feature dimension
            reverse = torch.flip(reverse, dims=[2])  # Flip along hidden_dim dimension

            # Average forward and reverse
            avg_token_embeddings = (forward + reverse) / 2  # (batch, seq_len, hidden_dim)

            # Compute both pooling strategies on the averaged embeddings
            mean_pooled = torch.mean(avg_token_embeddings, dim=1).to(torch.float32).cpu().numpy()
            max_pooled = torch.max(avg_token_embeddings, dim=1)[0].to(torch.float32).cpu().numpy()

            all_mean_embeddings.append(mean_pooled)
            all_max_embeddings.append(max_pooled)

    mean_embeddings = np.concatenate(all_mean_embeddings, axis=0)
    max_embeddings = np.concatenate(all_max_embeddings, axis=0)

    return {'mean': mean_embeddings, 'max': max_embeddings}


def train_xgboost_in_memory(X_train, y_train, X_val, y_val, output_model, pooling,
                           max_depth=6, learning_rate=0.1, n_estimators=1000, random_state=42):
    """Train XGBoost model on embeddings in memory."""
    import os
    import multiprocessing

    # Get number of CPU cores
    n_cores = multiprocessing.cpu_count()
    logging.info(f"Detected {n_cores} CPU cores")

    # Set OpenMP threads environment variable BEFORE training
    os.environ['OMP_NUM_THREADS'] = str(n_cores)

    logging.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    logging.info(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")

    # Set XGBoost parameters
    # Try GPU acceleration first, fall back to CPU if not available
    try:
        # Test if GPU is available for XGBoost
        test_gpu = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
        test_gpu.fit(X_train[:100], y_train[:100])
        tree_method = 'gpu_hist'
        device = 'cuda'
        logging.info("GPU detected! Using GPU acceleration (gpu_hist)")
    except Exception as e:
        tree_method = 'hist'
        device = 'cpu'
        logging.info(f"GPU not available, using CPU with {n_cores} threads: {e}")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'random_state': random_state,
        'tree_method': tree_method,
        'device': device,
    }

    # Add CPU-specific parameters if using CPU
    if device == 'cpu':
        params['n_jobs'] = n_cores
        params['nthread'] = n_cores

    # Train model
    logging.info(f"Training XGBoost model with params: {params}")
    if device == 'cpu':
        logging.info(f"Multi-threading: using {n_cores} CPU threads")
    else:
        logging.info(f"GPU acceleration: using {device}")

    # Check XGBoost build info
    logging.info(f"XGBoost version: {xgb.__version__}")

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    # Save model and validation data
    logging.info(f"Saving model to {output_model}...")
    model_data = {
        'model': model,
        'model_type': 'xgboost',
        'strategy': 'forward',
        'pooling': pooling,
        'X_val': X_val,
        'y_val': y_val
    }
    with open(output_model, 'wb') as f:
        pickle.dump(model_data, f)

    logging.info("Training complete!")


def train_nn_in_memory(X_train, y_train, X_val, y_val, output_model, pooling,
                      hidden_dim=256, num_hidden_layers=1, dropout=0.1, batch_size=64,
                      epochs=200, learning_rate=0.001, weight_decay=0.0, patience=20):
    """Train neural network model on embeddings in memory with early stopping."""
    logging.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    logging.info(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")

    # Create datasets and dataloaders
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    embedding_dim = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model = SimpleMLPClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout
    ).to(device)

    logging.info(f"Model architecture: {model}")
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    logging.info(f"\nTraining for up to {epochs} epochs with early stopping (patience={patience})...")
    best_val_auroc = 0.0
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(X_batch)

                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_dataset)
        val_auroc = roc_auc_score(val_labels, val_preds)
        val_auprc = average_precision_score(val_labels, val_preds)

        # Save best model based on validation AUROC and track early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            epochs_without_improvement = 0
            improvement_marker = " *"
        else:
            epochs_without_improvement += 1
            improvement_marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}{improvement_marker}")

        # Early stopping check
        if epochs_without_improvement >= patience:
            logging.info(f"\nEarly stopping triggered at epoch {epoch+1}")
            logging.info(f"No improvement in validation AUROC for {patience} epochs")
            logging.info(f"Best epoch: {best_epoch+1} with AUROC: {best_val_auroc:.4f}")
            break

    # Load best model based on validation performance
    model.load_state_dict(best_model_state)
    logging.info(f"\nTraining completed at epoch {epoch+1}")
    logging.info(f"Best Validation AUROC: {best_val_auroc:.4f} (epoch {best_epoch+1})")

    # Save model
    logging.info(f"Saving model to {output_model}...")
    model_data = {
        'model': model.cpu(),
        'model_state_dict': best_model_state,
        'model_type': 'neural_network',
        'strategy': 'forward',
        'pooling': pooling,
        'best_val_auroc': best_val_auroc,
        'best_epoch': best_epoch + 1,  # 1-indexed for display
        'total_epochs': epoch + 1,
        'config': {
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_hidden_layers': num_hidden_layers,
            'dropout': dropout,
            'patience': patience
        }
    }
    with open(output_model, 'wb') as f:
        pickle.dump(model_data, f)

    logging.info("Training complete!")


def evaluate_in_memory(model_data, X_test, y_test, output_metrics=None):
    """Evaluate model on test embeddings in memory."""
    model = model_data['model']
    model_type = model_data.get('model_type', 'xgboost')

    # Predictions
    logging.info("Generating predictions...")
    if model_type == 'neural_network':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        test_dataset = EmbeddingDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        y_pred_proba = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                y_pred_proba.extend(outputs.cpu().numpy())
        y_pred_proba = np.array(y_pred_proba)
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    # Calculate metrics
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print metrics
    logging.info("\n" + "="*50)
    logging.info("EVALUATION METRICS")
    logging.info("="*50)
    logging.info(f"AUROC:     {auroc:.4f}")
    logging.info(f"AUPRC:     {auprc:.4f}")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"F1 Score:  {f1:.4f}")
    logging.info(f"Pooling:   {model_data.get('pooling', 'N/A')}")
    logging.info(f"Test samples: {len(y_test)}")
    logging.info("="*50)

    # Save metrics
    if output_metrics:
        logging.info(f"\nSaving metrics to {output_metrics}...")
        with open(output_metrics, 'w') as f:
            f.write("="*50 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("="*50 + "\n")
            f.write(f"AUROC:     {auroc:.4f}\n")
            f.write(f"AUPRC:     {auprc:.4f}\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            f.write(f"Pooling:   {model_data.get('pooling', 'N/A')}\n")
            f.write(f"Test samples: {len(y_test)}\n")
            f.write("="*50 + "\n")

    return {'auroc': auroc, 'auprc': auprc, 'accuracy': accuracy, 'f1': f1}


def fetch_embeddings(input_tsv, output_pkl, model_dir, batch_size=32, layer_index=-1):
    """
    Fetch embeddings for sequences in input TSV file using PlantCAD2 model.

    Note: PlantCAD2 has built-in reverse complement equivariance, so no need to
    compute reverse complement embeddings separately. This function extracts both
    mean and max pooling embeddings in a single pass.

    Args:
        input_tsv: Path to input TSV file with columns: Chr, Start, End, Type, Species, Seq, Label
        output_pkl: Path to save embeddings as pickle file
        model_dir: Path to model directory
        batch_size: Batch size for processing sequences
        layer_index: Which hidden layer to use (-1 for last layer)
    """
    logging.info(f"Loading data from {input_tsv}...")
    df = pd.read_csv(input_tsv, sep='\t')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir, device)

    # Extract sequences
    sequences = df['Seq'].tolist()

    # Create dataloader
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    logging.info(f"Extracting embeddings for {len(df)} sequences with batch_size={batch_size}...")
    embeddings_dict = extract_embeddings(model, dataloader, device, layer_index)

    # Store in format compatible with train/evaluate functions
    # Save both mean and max pooling embeddings
    embeddings_data = {
        'mean': embeddings_dict['mean'],
        'max': embeddings_dict['max'],
        'labels': df['Label'].values,
        'metadata': df[['Chr', 'Start', 'End', 'Type', 'Species']].copy()
    }

    # Save embeddings
    logging.info(f"Saving embeddings to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(embeddings_data, f)

    logging.info(f"Done! Saved embeddings with shapes:")
    logging.info(f"  mean: {embeddings_data['mean'].shape}")
    logging.info(f"  max:  {embeddings_data['max'].shape}")


def main():
    parser = argparse.ArgumentParser(description='PlantCAD2 Embedding Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Fetch embeddings command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch embeddings from sequences (extracts both mean and max pooling)')
    fetch_parser.add_argument('--input', required=True, help='Input TSV file')
    fetch_parser.add_argument('--output', required=True, help='Output pickle file for embeddings')
    fetch_parser.add_argument('--model-dir', required=True, help='Path to PlantCAD2 model directory')
    fetch_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing sequences')
    fetch_parser.add_argument('--layer-index', type=int, default=-1, help='Which hidden layer to use (-1 for last)')

    # Train XGBoost command
    train_parser = subparsers.add_parser('train', help='Train XGBoost model')
    train_parser.add_argument('--train-embeddings', required=True, help='Training embeddings pickle file')
    train_parser.add_argument('--val-embeddings', required=True, help='Validation embeddings pickle file')
    train_parser.add_argument('--output', required=True, help='Output model pickle file')
    train_parser.add_argument('--pooling', choices=['mean', 'max'], default='mean',
                             help='Which pooling strategy to use (mean or max)')
    train_parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    train_parser.add_argument('--max-depth', type=int, default=6, help='XGBoost max depth')
    train_parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    train_parser.add_argument('--n-estimators', type=int, default=1000, help='Number of estimators')

    # Train Neural Network command
    train_nn_parser = subparsers.add_parser('train-nn', help='Train Neural Network model')
    train_nn_parser.add_argument('--train-embeddings', required=True, help='Training embeddings pickle file')
    train_nn_parser.add_argument('--val-embeddings', required=True, help='Validation embeddings pickle file')
    train_nn_parser.add_argument('--output', required=True, help='Output model pickle file')
    train_nn_parser.add_argument('--pooling', choices=['mean', 'max'], default='mean',
                                 help='Which pooling strategy to use (mean or max)')
    train_nn_parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension size')
    train_nn_parser.add_argument('--num-hidden-layers', type=int, default=1,
                                 help='Total number of linear layers (1=single linear, 2=one hidden layer, etc.)')
    train_nn_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    train_nn_parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    train_nn_parser.add_argument('--epochs', type=int, default=200, help='Maximum number of training epochs')
    train_nn_parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs without improvement)')
    train_nn_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_nn_parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization weight')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', required=True, help='Trained model pickle file')
    eval_parser.add_argument('--test-embeddings', required=True, help='Test embeddings pickle file')
    eval_parser.add_argument('--pooling', choices=['mean', 'max'],
                            help='Which pooling strategy to use (default: use pooling from training)')
    eval_parser.add_argument('--output', help='Output metrics text file')

    args = parser.parse_args()

    if args.command == 'fetch':
        fetch_embeddings(
            input_tsv=args.input,
            output_pkl=args.output,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
            layer_index=args.layer_index
        )
    elif args.command == 'train':
        # Load train and val embeddings and select pooling strategy
        logging.info(f"Loading training embeddings from {args.train_embeddings}...")
        with open(args.train_embeddings, 'rb') as f:
            train_data = pickle.load(f)

        logging.info(f"Loading validation embeddings from {args.val_embeddings}...")
        with open(args.val_embeddings, 'rb') as f:
            val_data = pickle.load(f)

        X_train = train_data[args.pooling]
        y_train = train_data['labels']
        X_val = val_data[args.pooling]
        y_val = val_data['labels']

        logging.info(f"Using {args.pooling} pooling strategy")
        train_xgboost_in_memory(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_model=args.output,
            pooling=args.pooling,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            random_state=args.random_state
        )

    elif args.command == 'train-nn':
        # Load train and val embeddings and select pooling strategy
        logging.info(f"Loading training embeddings from {args.train_embeddings}...")
        with open(args.train_embeddings, 'rb') as f:
            train_data = pickle.load(f)

        logging.info(f"Loading validation embeddings from {args.val_embeddings}...")
        with open(args.val_embeddings, 'rb') as f:
            val_data = pickle.load(f)

        X_train = train_data[args.pooling]
        y_train = train_data['labels']
        X_val = val_data[args.pooling]
        y_val = val_data['labels']

        logging.info(f"Using {args.pooling} pooling strategy")
        train_nn_in_memory(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            output_model=args.output,
            pooling=args.pooling,
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience
        )

    elif args.command == 'evaluate':
        # Load model
        logging.info(f"Loading model from {args.model}...")
        with open(args.model, 'rb') as f:
            model_data = pickle.load(f)

        pooling = args.pooling if args.pooling else model_data.get('pooling', 'mean')
        logging.info(f"Using pooling strategy: {pooling}")

        # Load test embeddings
        logging.info(f"Loading test embeddings from {args.test_embeddings}...")
        with open(args.test_embeddings, 'rb') as f:
            test_data = pickle.load(f)

        X_test = test_data[pooling]
        y_test = test_data['labels']

        evaluate_in_memory(
            model_data=model_data,
            X_test=X_test,
            y_test=y_test,
            output_metrics=args.output
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
