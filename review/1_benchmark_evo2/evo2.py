from evo2 import Evo2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import xgboost as xgb


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


def reverse_complement(seq):
    """Generate reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
                  'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
                  'N': 'N', 'n': 'n'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def get_final_token_embedding(sequence, model, layer_name):
    """Extract final token embedding from Evo2 model for a single sequence."""
    input_ids = torch.tensor(
        model.tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to('cuda:0')
    with torch.no_grad():
        _, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
    return embeddings[layer_name][0, -1, :].cpu().to(torch.float32).numpy()  # shape: (hidden_dim,)


def get_batched_embeddings(sequences, model, layer_name, batch_size=32):
    """Extract final token embeddings for a batch of sequences."""
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch_seqs = sequences[i:i + batch_size]

        # Tokenize all sequences in batch
        tokenized = [model.tokenizer.tokenize(seq) for seq in batch_seqs]

        # Find max length in batch
        max_len = max(len(t) for t in tokenized)

        # Pad sequences and create attention mask
        batch_input_ids = []
        seq_lengths = []
        for tokens in tokenized:
            seq_lengths.append(len(tokens))
            # Pad with 0s (assuming 0 is padding token)
            padded = tokens + [0] * (max_len - len(tokens))
            batch_input_ids.append(padded)

        # Convert to tensor
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.int).to('cuda:0')

        # Get embeddings
        with torch.no_grad():
            _, embeddings = model(batch_input_ids, return_embeddings=True, layer_names=[layer_name])

        # Extract final token embedding for each sequence (before padding)
        batch_embs = embeddings[layer_name]  # shape: (batch_size, seq_len, hidden_dim)
        for j, seq_len in enumerate(seq_lengths):
            # Get the last real token (before padding)
            emb = batch_embs[j, seq_len - 1, :].cpu().to(torch.float32).numpy()
            all_embeddings.append(emb)

    return np.array(all_embeddings)


def fetch_embeddings(input_tsv, output_pkl, use_reverse_complement=True, layer_name='blocks.26', batch_size=32):
    """
    Fetch Evo2 embeddings for sequences in input TSV file.

    Args:
        input_tsv: Path to input TSV file with columns: Chr, Start, End, Type, Species, Seq, Label
        output_pkl: Path to save embeddings as pickle file
        use_reverse_complement: Whether to compute reverse complement embeddings
        layer_name: Which layer to extract embeddings from
        batch_size: Batch size for processing sequences (default: 32)
    """
    print(f"Loading data from {input_tsv}...")
    df = pd.read_csv(input_tsv, sep='\t')

    print(f"Loading Evo2 model...")
    evo2_model = Evo2('evo2_7b_base')

    # Extract sequences
    forward_seqs = df['Seq'].tolist()

    print(f"Extracting forward embeddings for {len(df)} sequences with batch_size={batch_size}...")
    forward_embeddings = get_batched_embeddings(forward_seqs, evo2_model, layer_name, batch_size)

    embeddings_data = {
        'forward': forward_embeddings,
        'reverse': None,
        'labels': df['Label'].values,
        'metadata': df[['Chr', 'Start', 'End', 'Type', 'Species']].copy()
    }

    # Reverse complement embeddings
    if use_reverse_complement:
        print(f"Extracting reverse complement embeddings...")
        reverse_seqs = [reverse_complement(seq) for seq in forward_seqs]
        reverse_embeddings = get_batched_embeddings(reverse_seqs, evo2_model, layer_name, batch_size)
        embeddings_data['reverse'] = reverse_embeddings

    # Save embeddings
    print(f"Saving embeddings to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(embeddings_data, f)

    print(f"Done! Saved embeddings with shape: forward={embeddings_data['forward'].shape}")
    if use_reverse_complement:
        print(f"  reverse={embeddings_data['reverse'].shape}")


def train_nn(train_embeddings_pkl, val_embeddings_pkl, output_model, strategy='concatenate',
             hidden_dim=256, num_hidden_layers=1, dropout=0.1, batch_size=64,
             epochs=200, learning_rate=0.001, weight_decay=0.0, patience=20):
    """
    Train neural network model on embeddings with early stopping.

    Args:
        train_embeddings_pkl: Path to pickle file with training embeddings
        val_embeddings_pkl: Path to pickle file with validation embeddings
        output_model: Path to save trained model
        strategy: How to use embeddings - 'forward', 'reverse', 'average', 'concatenate'
        hidden_dim: Hidden dimension size
        num_hidden_layers: Number of hidden layers
        dropout: Dropout rate
        batch_size: Training batch size
        epochs: Maximum number of training epochs
        learning_rate: Learning rate
        weight_decay: L2 regularization weight
        patience: Early stopping patience (epochs without improvement)
    """
    print(f"Loading training embeddings from {train_embeddings_pkl}...")
    with open(train_embeddings_pkl, 'rb') as f:
        train_data = pickle.load(f)

    print(f"Loading validation embeddings from {val_embeddings_pkl}...")
    with open(val_embeddings_pkl, 'rb') as f:
        val_data = pickle.load(f)

    # Prepare features based on strategy for training data
    print(f"Preparing features using strategy: {strategy}")

    def prepare_features(data, strategy):
        if strategy == 'forward':
            return data['forward']
        elif strategy == 'reverse':
            if data['reverse'] is None:
                raise ValueError("Reverse embeddings not available. Re-run fetch_embeddings with use_reverse_complement=True")
            return data['reverse']
        elif strategy == 'average':
            if data['reverse'] is None:
                raise ValueError("Reverse embeddings not available. Re-run fetch_embeddings with use_reverse_complement=True")
            return (data['forward'] + data['reverse']) / 2
        elif strategy == 'concatenate':
            if data['reverse'] is None:
                raise ValueError("Reverse embeddings not available. Re-run fetch_embeddings with use_reverse_complement=True")
            return np.concatenate([data['forward'], data['reverse']], axis=1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from: forward, reverse, average, concatenate")

    X_train = prepare_features(train_data, strategy)
    y_train = train_data['labels']

    X_val = prepare_features(val_data, strategy)
    y_val = val_data['labels']

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")

    # Create datasets and dataloaders
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    embedding_dim = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SimpleMLPClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout
    ).to(device)

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    print(f"\nTraining for up to {epochs} epochs with early stopping (patience={patience})...")
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
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}{improvement_marker}")

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"No improvement in validation AUROC for {patience} epochs")
            print(f"Best epoch: {best_epoch+1} with AUROC: {best_val_auroc:.4f}")
            break

    # Load best model based on validation performance
    model.load_state_dict(best_model_state)
    print(f"\nTraining completed at epoch {epoch+1}")
    print(f"Best Validation AUROC: {best_val_auroc:.4f} (epoch {best_epoch+1})")

    # Save model
    print(f"Saving model to {output_model}...")
    model_data = {
        'model': model.cpu(),  # Move to CPU for saving
        'model_state_dict': best_model_state,
        'model_type': 'neural_network',
        'strategy': strategy,
        'best_val_auroc': best_val_auroc,
        'best_epoch': best_epoch + 1,  # 1-indexed for display
        'total_epochs': epoch + 1,
        'train_metadata': train_data['metadata'],
        'val_metadata': val_data['metadata'],
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

    print("Training complete!")
    return model


def evaluate(model_pkl, test_embeddings_pkl, strategy=None, output_metrics=None):
    """
    Evaluate trained model (XGBoost or Neural Network) on test embeddings.

    Args:
        model_pkl: Path to trained model pickle file
        test_embeddings_pkl: Path to test embeddings pickle file
        strategy: Embedding strategy (if None, use the one from training)
        output_metrics: Optional path to save metrics as text file
    """
    print(f"Loading model from {model_pkl}...")
    with open(model_pkl, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    model_type = model_data.get('model_type', 'xgboost')  # Default to xgboost for backward compatibility

    # Force XGBoost to use CPU for prediction to avoid GPU memory issues
    if model_type == 'xgboost':
        import json
        import tempfile
        try:
            # Create a new model with CPU configuration
            print("Reconfiguring XGBoost model to use CPU...")
            old_booster = model.get_booster()

            # Save the booster to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_path = f.name
                old_booster.save_model(temp_path)

            # Create new booster with CPU device
            new_booster = xgb.Booster({'device': 'cpu'})
            new_booster.load_model(temp_path)

            # Replace the model's booster
            model._Booster = new_booster

            # Clean up
            import os
            os.unlink(temp_path)
            print("Successfully configured XGBoost to use CPU")
        except Exception as e:
            print(f"Warning: Could not reconfigure to CPU: {e}")
            print("Predictions may fail due to GPU memory constraints")

    # Use strategy from training if not specified
    if strategy is None:
        strategy = model_data.get('strategy', 'concatenate')
    print(f"Using strategy: {strategy}")

    # Load test embeddings
    print(f"Loading test embeddings from {test_embeddings_pkl}...")
    with open(test_embeddings_pkl, 'rb') as f:
        test_data = pickle.load(f)

    # Prepare features
    def prepare_features(data, strategy):
        if strategy == 'forward':
            return data['forward']
        elif strategy == 'reverse':
            if data['reverse'] is None:
                raise ValueError("Reverse embeddings not available.")
            return data['reverse']
        elif strategy == 'average':
            if data['reverse'] is None:
                raise ValueError("Reverse embeddings not available.")
            return (data['forward'] + data['reverse']) / 2
        elif strategy == 'concatenate':
            if data['reverse'] is None:
                raise ValueError("Reverse embeddings not available.")
            return np.concatenate([data['forward'], data['reverse']], axis=1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    X_test = prepare_features(test_data, strategy)
    y_test = test_data['labels']

    # Predictions
    print("Generating predictions...")
    if model_type == 'neural_network':
        # Neural network prediction
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
        # XGBoost prediction
        # Check if X_test is stored in model_data (old format)
        if 'X_test' in model_data and 'y_test' in model_data:
            X_test = model_data['X_test']
            y_test = model_data['y_test']

        # Batch predictions to avoid GPU memory issues
        batch_size = 10000
        y_pred_proba_list = []
        y_pred_list = []

        print(f"Predicting in batches of {batch_size} to avoid memory issues...")
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            # Use predict_proba on batches to get proper probabilities
            proba_batch = model.predict_proba(X_batch)
            y_pred_proba_list.append(proba_batch[:, 1])
            y_pred_list.append(model.predict(X_batch))
            print(f"  Processed {min(i+batch_size, len(X_test))}/{len(X_test)} samples")

        y_pred_proba = np.concatenate(y_pred_proba_list)
        y_pred = np.concatenate(y_pred_list)

    # Calculate metrics
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"AUROC:     {auroc:.4f}")
    print(f"AUPRC:     {auprc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Strategy:  {model_data['strategy']}")
    print(f"Test samples: {len(y_test)}")
    print("="*50)

    # Save metrics
    if output_metrics:
        print(f"\nSaving metrics to {output_metrics}...")
        with open(output_metrics, 'w') as f:
            f.write("="*50 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("="*50 + "\n")
            f.write(f"AUROC:     {auroc:.4f}\n")
            f.write(f"AUPRC:     {auprc:.4f}\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            f.write(f"Strategy:  {model_data['strategy']}\n")
            f.write(f"Test samples: {len(y_test)}\n")
            f.write("="*50 + "\n")

    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Evo2 Embedding Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Fetch embeddings command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch embeddings from sequences')
    fetch_parser.add_argument('--input', required=True, help='Input TSV file')
    fetch_parser.add_argument('--output', required=True, help='Output pickle file for embeddings')
    fetch_parser.add_argument('--no-reverse', action='store_true', help='Do not compute reverse complement')
    fetch_parser.add_argument('--layer', default='blocks.26', help='Layer to extract embeddings from')
    fetch_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing sequences')

    # Train XGBoost command
    train_parser = subparsers.add_parser('train', help='Train XGBoost model')
    train_parser.add_argument('--train-embeddings', required=True, help='Training embeddings pickle file')
    train_parser.add_argument('--val-embeddings', required=True, help='Validation embeddings pickle file')
    train_parser.add_argument('--output', required=True, help='Output model pickle file')
    train_parser.add_argument('--strategy', choices=['forward', 'reverse', 'average', 'concatenate'],
                             default='concatenate', help='Embedding strategy')
    train_parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    train_parser.add_argument('--max-depth', type=int, default=6, help='XGBoost max depth')
    train_parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    train_parser.add_argument('--n-estimators', type=int, default=1000, help='Number of estimators')

    # Train Neural Network command
    train_nn_parser = subparsers.add_parser('train-nn', help='Train Neural Network model')
    train_nn_parser.add_argument('--train-embeddings', required=True, help='Training embeddings pickle file')
    train_nn_parser.add_argument('--val-embeddings', required=True, help='Validation embeddings pickle file')
    train_nn_parser.add_argument('--output', required=True, help='Output model pickle file')
    train_nn_parser.add_argument('--strategy', choices=['forward', 'reverse', 'average', 'concatenate'],
                                 default='concatenate', help='Embedding strategy')
    train_nn_parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension size')
    train_nn_parser.add_argument('--num-hidden-layers', type=int, default=1, help='Total number of linear layers (1=single linear, 2=one hidden layer, 3=two hidden layers, etc.)')
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
    eval_parser.add_argument('--strategy', choices=['forward', 'reverse', 'average', 'concatenate'],
                            help='Embedding strategy (default: use strategy from training)')
    eval_parser.add_argument('--output', help='Output metrics text file')

    args = parser.parse_args()

    if args.command == 'fetch':
        fetch_embeddings(
            input_tsv=args.input,
            output_pkl=args.output,
            use_reverse_complement=not args.no_reverse,
            layer_name=args.layer,
            batch_size=args.batch_size
        )
    elif args.command == 'train':
        # Load train and val embeddings and select strategy
        print(f"Loading training embeddings from {args.train_embeddings}...")
        with open(args.train_embeddings, 'rb') as f:
            train_data = pickle.load(f)

        print(f"Loading validation embeddings from {args.val_embeddings}...")
        with open(args.val_embeddings, 'rb') as f:
            val_data = pickle.load(f)

        # Prepare features based on strategy
        def prepare_features(data, strategy):
            if strategy == 'forward':
                return data['forward']
            elif strategy == 'reverse':
                if data['reverse'] is None:
                    raise ValueError("Reverse embeddings not available. Re-run fetch_embeddings with use_reverse_complement=True")
                return data['reverse']
            elif strategy == 'average':
                if data['reverse'] is None:
                    raise ValueError("Reverse embeddings not available. Re-run fetch_embeddings with use_reverse_complement=True")
                return (data['forward'] + data['reverse']) / 2
            elif strategy == 'concatenate':
                if data['reverse'] is None:
                    raise ValueError("Reverse embeddings not available. Re-run fetch_embeddings with use_reverse_complement=True")
                return np.concatenate([data['forward'], data['reverse']], axis=1)
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Choose from: forward, reverse, average, concatenate")

        X_train = prepare_features(train_data, args.strategy)
        y_train = train_data['labels']
        X_val = prepare_features(val_data, args.strategy)
        y_val = val_data['labels']

        print(f"Using {args.strategy} strategy")

        # Train XGBoost model
        import os
        import multiprocessing

        # Get number of CPU cores
        n_cores = multiprocessing.cpu_count()
        print(f"Detected {n_cores} CPU cores")

        # Set OpenMP threads environment variable BEFORE training
        os.environ['OMP_NUM_THREADS'] = str(n_cores)

        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")

        # Set XGBoost parameters
        # Try GPU acceleration first, fall back to CPU if not available
        try:
            # Test if GPU is available for XGBoost
            test_gpu = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
            test_gpu.fit(X_train[:100], y_train[:100])
            tree_method = 'gpu_hist'
            device = 'cuda'
            print("GPU detected! Using GPU acceleration (gpu_hist)")
        except Exception as e:
            tree_method = 'hist'
            device = 'cpu'
            print(f"GPU not available, using CPU with {n_cores} threads: {e}")

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'random_state': args.random_state,
            'tree_method': tree_method,
            'device': device,
        }

        # Add CPU-specific parameters if using CPU
        if device == 'cpu':
            params['n_jobs'] = n_cores // 2  # Use half of available cores
            params['nthread'] = n_cores // 2

        # Train model
        print(f"Training XGBoost model with params: {params}")
        if device == 'cpu':
            print(f"Multi-threading: using {n_cores} CPU threads")
        else:
            print(f"GPU acceleration: using {device}")

        # Check XGBoost build info
        print(f"XGBoost version: {xgb.__version__}")

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )

        # Save model and validation data
        print(f"Saving model to {args.output}...")
        model_data = {
            'model': model,
            'strategy': args.strategy,
            'X_val': X_val,
            'y_val': y_val,
            'metadata': train_data['metadata']
        }
        with open(args.output, 'wb') as f:
            pickle.dump(model_data, f)

        print("Training complete!")
    elif args.command == 'train-nn':
        train_nn(
            train_embeddings_pkl=args.train_embeddings,
            val_embeddings_pkl=args.val_embeddings,
            output_model=args.output,
            strategy=args.strategy,
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
        evaluate(
            model_pkl=args.model,
            test_embeddings_pkl=args.test_embeddings,
            strategy=args.strategy,
            output_metrics=args.output
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
