import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn import metrics
import wandb
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from Bio import SeqIO
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import fire
import os
from typing import List, Tuple, Optional
from tqdm import tqdm


class DNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        # remove self.one_hot_encoded

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].upper()
        base_to_index = {'A':0,'T':1,'C':2,'G':3}
        arr = np.zeros((4, len(seq))) 
        for i, b in enumerate(seq):
            if b in base_to_index:
                arr[base_to_index[b], i] = 1
        x = torch.from_numpy(arr).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class DanQLightning(pl.LightningModule):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters()
        self._is_classifier = True
        self.learning_rate = learning_rate

        self.validation_step_outputs = []

        self.conv1 = torch.nn.Conv1d(
            in_channels=4,
            out_channels=320,
            kernel_size=26
        )
        self.maxpool1 = torch.nn.MaxPool1d(
            kernel_size=13,
            stride=13,
            ceil_mode=True
        )
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.lstm = torch.nn.LSTM(
            input_size=320,
            hidden_size=320,
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.linear1 = torch.nn.Linear(
            in_features=640,
            out_features=1
        )

    @property
    def is_classifier(self) -> bool:
        return self._is_classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        output, (hn, cn) = self.lstm(x.transpose(-1, -2))
        x = hn.transpose(0, -2).flatten(-2, -1)
        x = self.dropout2(x)

        if self.is_classifier:
            return torch.nn.functional.sigmoid(self.linear1(x))

        return torch.nn.functional.relu(self.linear1(x))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        
        # Calculate metrics
        with torch.no_grad():
            y_cpu = y.cpu().numpy()
            y_hat_cpu = y_hat.cpu().numpy()
            try:
                accuracy = accuracy_score(y_cpu, (y_hat_cpu >= 0.5).astype(int))
                f1 = metrics.f1_score(y_cpu, (y_hat_cpu >= 0.5).astype(int))
                roc_auc = roc_auc_score(y_cpu, y_hat_cpu)
                precision, recall, _ = precision_recall_curve(y_cpu, y_hat_cpu)
                average_precision = auc(recall, precision)
                balance = np.mean(y_cpu)

                self.log('train/accuracy', accuracy, prog_bar=True)
                self.log('train/f1', f1, prog_bar=True)
                self.log('train/roc_auc', roc_auc, prog_bar=True)
                self.log('train/average_precision', average_precision, prog_bar=True)
                self.log('train/balance', balance, prog_bar=True)
            except:
                pass  # Handle case where batch has only one class
                
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.functional.binary_cross_entropy(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        # Store outputs for use in on_validation_epoch_end
        output = {'val_loss': loss, 'y': y, 'y_hat': y_hat}
        self.validation_step_outputs.append(output)
        return output
    
    def on_validation_epoch_end(self):
        # Access the saved outputs
        outputs = self.validation_step_outputs
        
        # Aggregate all validation predictions
        y_true = torch.cat([output['y'] for output in outputs]).cpu().numpy()
        y_pred = torch.cat([output['y_hat'] for output in outputs]).cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        f1 = metrics.f1_score(y_true, (y_pred >= 0.5).astype(int))
        roc_auc = roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        average_precision = auc(recall, precision)
        balance = np.mean(y_true)
        
        # Log metrics
        self.log('eval/accuracy', accuracy, prog_bar=True)
        self.log('eval/f1', f1, prog_bar=True)
        self.log('eval/roc_auc', roc_auc, prog_bar=True)
        self.log('eval/average_precision', average_precision, prog_bar=True)
        self.log('eval/balance', balance, prog_bar=True)
        
        # Clear the outputs list for the next epoch
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train(
    train: str,
    valid: str,
    output_dir: str,
    run_id: str = None,
    balance: bool = False,
    device: str = 'cuda:0',
    batch_size: int = 2048,
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    use_wandb: bool = True,
    wandb_project: str = "PlantCAD2",
    wandb_entity: str = "jingjingzhai",
):
    """Train a DanQ model on DNA sequences.
    
    Args:
        train: Path to training sequences in TSV format
        valid: Path to validation sequences in TSV format
        output_dir: Directory to save model checkpoints
        run_id: Unique ID for this run (used for wandb)
        balance: Whether to balance the classes in the training data
        device: Device to use for training ('cuda:0', 'cuda:1', etc.)
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimization
        use_wandb: Whether to log metrics to Weights & Biases
        wandb_project: Project name for Weights & Biases
        wandb_entity: Entity name for Weights & Biases
    
    Returns:
        Path to the best model checkpoint
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up WandB logging
    config = {
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
    }
    
    if use_wandb:
        wandb_logger = WandbLogger(
            project=wandb_project, 
            id=run_id, 
            resume='allow',
            entity=wandb_entity, 
            config=config
        )
    else:
        wandb_logger = None
    
    # Prepare and one-hot encode DNA sequences
    trainDATA = pd.read_csv(train, delimiter='\t')
    validDATA = pd.read_csv(valid, delimiter='\t')

    trainSeq = trainDATA['Seq'].values
    validSeq = validDATA['Seq'].values

    trainLabel = trainDATA['Label'].values
    validLabel = validDATA['Label'].values
    

    # Handle class imbalance if requested
    if balance:
        trainPosSeq = [trainSeq[i] for i in range(len(trainSeq)) if trainLabel[i] == 1]
        trainNegSeq = [trainSeq[i] for i in range(len(trainSeq)) if trainLabel[i] == 0]
        # Shuffle and downsample and set seed
        np.random.seed(1024)
        np.random.shuffle(trainNegSeq)
        trainNegSeq = trainNegSeq[:len(trainPosSeq)]
        trainSeq = trainPosSeq + trainNegSeq
        trainLabel = [1]*len(trainPosSeq) + [0]*len(trainNegSeq)
        train_dataset = DNADataset(sequences=trainSeq, labels=trainLabel)
    else:
        train_dataset = DNADataset(sequences=trainSeq, labels=trainLabel)

    val_dataset = DNADataset(sequences=validSeq, labels=validLabel)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = DanQLightning(learning_rate=learning_rate)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='bestmodel',
        save_top_k=1,
        verbose=True,
        monitor='eval/average_precision',
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='eval/average_precision',
        patience=20,
        min_delta=0.001,
        verbose=True,
        mode='max'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if 'cuda' in device else 'cpu',
        devices=[int(device.split(':')[-1])] if 'cuda' in device else None,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        check_val_every_n_epoch=1,  # Check validation at the end of every epoch
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Print best model path
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path


def evaluate(
    model_path: str,
    test: str,
    device: str = 'cuda:0',
    batch_size: int = 2048,
    output_file: Optional[str] = None
):
    """Evaluate a trained DanQ model on test data.
    
    Args:
        model_path: Path to the trained model checkpoint
        test: Path to test sequences in TSV format
        device: Device to use for evaluation
        batch_size: Batch size for evaluation
        output_file: Path to save evaluation results (optional)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load test data
    testDATA = pd.read_csv(test, delimiter='\t')
    testSeq = testDATA['Seq'].values
    testLabel = testDATA['Label'].values

    
    # Create dataset and dataloader
    test_dataset = DNADataset(sequences=testSeq, labels=testLabel, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load model
    model = DanQLightning.load_from_checkpoint(model_path)
    model.eval()
    
    # Initialize trainer for evaluation
    trainer = pl.Trainer(
        accelerator='gpu' if 'cuda' in device else 'cpu',
        devices=[int(device.split(':')[-1])] if 'cuda' in device else None,
        logger=False
    )
    
    # Evaluate model
    results = trainer.test(model, test_loader)[0]
    
    # Manual calculation of metrics
    all_preds = []
    all_labels = []
    
    model.to(torch.device(device))
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(torch.device(device))
            y_hat = model(x).squeeze().cpu().numpy()
            all_preds.extend(y_hat)
            all_labels.extend(y.numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    auc_score = roc_auc_score(all_labels, all_preds)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)
    
    # Binary predictions using 0.5 threshold
    binary_preds = (all_preds >= 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    
    metrics_dict = {
        'test_auc': auc_score,
        'test_prauc': pr_auc,
        'test_accuracy': accuracy
    }
    
    # Print metrics
    print(f"AUC: {auc_score:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save results if output file is provided
    if output_file:
        results_df = pd.DataFrame({
            'sequence_id': testName,
            'true_label': all_labels,
            'prediction': all_preds,
            'binary_prediction': binary_preds
        })
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return metrics_dict


def predict(
    model_path: str,
    test: str,
    output_file: str,
    device: str = 'cuda:0',
    batch_size: int = 2048
):
    """Generate predictions for sequences without known labels.
    
    Args:
        model_path: Path to the trained model checkpoint
        test: Path to input sequences in TSV format
        output_file: Path to save predictions
        device: Device to use for prediction
        batch_size: Batch size for prediction
    
    Returns:
        Path to the output file with predictions
    """
    testDATA = pd.read_csv(test, delimiter='\t')
    sequences = testDATA['Seq'].values
    
    # Create dummy labels (will not be used)
    dummy_labels = np.zeros(len(sequences))
    
    # Create dataset and dataloader
    input_dataset = DNADataset(sequences=sequences, labels=dummy_labels)
    input_loader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load model
    model = DanQLightning.load_from_checkpoint(model_path)
    model.eval()
    model.to(torch.device(device))
    
    # Generate predictions
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(input_loader, desc="Generating Predictions"):
            x, _ = batch
            x = x.to(torch.device(device))
            y_hat = model(x).squeeze().cpu().numpy()
            all_preds.extend(y_hat)
    
    # Convert to numpy array
    all_preds = np.array(all_preds)
    
    # Save predictions
    results_df = pd.DataFrame({
        'probability_positive': all_preds,
    })
    results_df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")
    return output_file


if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'evaluate': evaluate,
        'predict': predict
    })