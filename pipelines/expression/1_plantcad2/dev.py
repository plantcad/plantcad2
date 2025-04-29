import os
import fire
import torch
import logging
import numpy as np
import multiprocessing
from pathlib import Path
from typing import Optional
from datasets import Dataset, Features, Value

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import pandas as pd
from scipy.stats import spearmanr, pearsonr


logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# CLI Functions
# ------------------------------------------------------------------------------

def tokenize(
    data_dir: str,
    output_path: Optional[str] = None,
    model_name: str = None,
    sequence_length: int = 8192,
    batch_size: int = 1000,
    max_batches: Optional[int] = None,
    num_proc: Optional[int] = None,
) -> None:
    """Tokenize sequences from a TSV file and save as compressedparquet format.

    Processes gene sequences from a tab-separated file and converts them into tokenized 
    format suitable for model training. The input TSV must have columns ['Gene', 'Seq', 'Label'].
    All sequences must be of the specified length.

    Parameters
    ----------
    data_dir : str
        Path to input TSV file containing sequences to tokenize
    output_path : str, optional
        Path to save tokenized output. If not provided, will use input path with .parquet extension
    model_name : str, optional
        Name or path of the model whose tokenizer should be used
    sequence_length : int, optional
        Expected length of all input sequences, by default 8192
    batch_size : int, optional
        Number of sequences to process in each batch, by default 1000
    max_batches : int, optional
        If provided, limit processing to this many batches (for testing)
    num_proc : int, optional
        Number of processes to use. Defaults to CPU count if not specified

    Examples
    --------
    >>> python main.py tokenize --data_dir=$DATA_DIR/train.tsv
    >>> python main.py tokenize --data_dir=$DATA_DIR/valid.tsv --max_batches=10  # For testing
    """
    if output_path is None:
        output_path = str(Path(data_dir).with_suffix('.parquet'))
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()
    
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create dataset from CSV with streaming
    dataset = Dataset.from_csv(
        data_dir, 
        sep='\t'
    )

    for c in list(dataset.column_names):
        new_name = c.lower()
        if c != new_name:
            dataset = dataset.rename_column(c, new_name)

    # Limit dataset size if max_batches is specified
    if max_batches is not None:
        max_examples = max_batches * batch_size
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        logger.info(f"Limited dataset to {max_examples} examples ({max_batches} batches)")

    def tokenize_batch(examples):
        
        # Tokenize the batch
        tokenized = tokenizer(
            [str(seq) for seq in examples['seq']],
            padding='max_length',
            truncation=True,
            max_length=sequence_length,
            # This is crucial to avoid EOS tokens being suffixed to sequences
            # and extending them beyond the expected length `sequence_length`
            add_special_tokens=False,
        )

        # Validate sequence lengths
        lengths = set(map(len, tokenized["input_ids"]))
        if lengths != {sequence_length}:
            raise ValueError(
                f"All sequences must be of length {sequence_length}; "
                f"found sequence batch with lengths {lengths}"
            )
        
        return {'input_ids': tokenized['input_ids']}

    # Process in batches and save
    logger.info(f"Tokenizing sequences in batches of {batch_size}")
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=["seq"],
        num_proc=num_proc,
    )

    logger.info(f"Saving tokenized dataset to {output_path}")
    # See options at https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html
    tokenized_dataset.to_parquet(output_path, compression="zstd")
    logger.info("Tokenization complete")

def display(model_name: str = None) -> None:
    """Display the structure and parameters of the model with LoRA adapter.

    Creates a LoRA-adapted model and prints detailed information about its structure
    and parameters, including which parameters are trainable under LoRA.

    Parameters
    ----------
    model_name : str, optional
        Name or path of the base model to analyze

    Examples
    --------
    >>> python main.py display
    """
    logger.info(f"Loading base model from {model_name}")
    base_model = load_base_model(model_name)

    logger.info(f"Configuring LoRA adapter")
    model = create_peft_model(base_model)
    
    logger.info(f"Model structure:\n{model}")


    logger.info(f"Model parameters:")
    parameters = []
    for n, p in model.named_parameters():
        parameters.append(dict(
            name=n,
            is_trainable=p.requires_grad,
            shape=tuple(p.shape) if hasattr(p, 'shape') else None,
            size=p.numel(),
        ))
    
    # Calculate column widths based on data and headers
    col_widths = {
        'name': max(len("Name"), max(len(str(p['name'])) for p in parameters)),
        'is_trainable': max(len("Trainable"), max(len(str(p['is_trainable'])) for p in parameters)),
        'shape': max(len("Shape"), max(len(str(p['shape'])) for p in parameters)),
        'size': max(len("Size"), max(len(str(p['size'])) for p in parameters))
    }

    # Add some padding
    col_widths = {k: v + 2 for k, v in col_widths.items()}

    # Print header
    header = (f"{'Name':<{col_widths['name']}} "
             f"{'Trainable':<{col_widths['is_trainable']}} "
             f"{'Shape':<{col_widths['shape']}} "
             f"{'Size':<{col_widths['size']}}")
    logger.info(header)
    logger.info('-' * (sum(col_widths.values()) + len(col_widths) - 1))  # Separator line
    
    # Print each parameter row
    for param in parameters:
        row = (f"{param['name']:<{col_widths['name']}} "
               f"{str(param['is_trainable']):<{col_widths['is_trainable']}} "
               f"{str(param['shape']):<{col_widths['shape']}} "
               f"{param['size']:<{col_widths['size']}}")
        logger.info(row)

    logger.info("Model display complete")

def train(
    train_dir: str,
    valid_dir: str,
    output_dir: str = "/tmp/pcv2-ft",
    model_name: str = None,
    task_type: str = "classification",
    train_batch_size: int = 3,
    eval_batch_size: int = 3,
    eval_num_samples: Optional[int] = 0,
    max_steps: int = 500,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "pcv2-ft",
    wandb_run_name: Optional[str] = None,
    learning_rate: float = 1e-3,
    warmup_steps: int = 50,
    lr_scheduler_type: str = "linear",
    gradient_accumulation_steps: int = 64,
    bf16: bool = True,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    eval_strategy: str = "steps",
    eval_steps: int = 0,
    save_strategy: str = "steps",
    save_steps: int = 100,
    logging_steps: int = 100,
    remove_unused_columns: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    crop_length: int = 0
) -> None:
    """Train a LoRA adapter on top of a base model for sequence classification.

    Performs fine-tuning using Low-Rank Adaptation (LoRA) on a pre-trained model.
    Uses an 80/20 train/validation split of the training data for monitoring.
    With default settings, approximately 1.4% of model parameters are trainable.

    Parameters
    ----------
    train_dir : str
        Directory containing tokenized training data (train_tokenize_TE_2k.parquet)
    valid_dir : str
        Directory containing tokenized validation data (valid_tokenize_TE_2k.parquet)
    output_dir : str, optional
        Directory to save model checkpoints and training outputs
    model_name : str, optional
        Name or path of base model to fine-tune
    task_type : str, optional
        Type of task, either "classification" or "regression", by default "classification"
    train_batch_size : int, optional
        Batch size for training, by default 3. Each sequence uses ~11GB GPU memory
    eval_batch_size : int, optional
        Batch size for evaluation, by default 3
    eval_num_samples : int, optional
        Number of samples to use for validation, by default 128
    max_steps : int, optional
        Maximum number of training steps, by default 500
    seed : int, optional
        Random seed for reproducibility, by default 42
    use_wandb : bool, optional
        Whether to log metrics to Weights & Biases, by default False
    wandb_project : str, optional
        W&B project name if use_wandb=True, by default "pcv2-ft"
    wandb_run_name : str, optional
        W&B run name if use_wandb=True
    learning_rate : float, optional
        Learning rate for training, by default 1e-3
    warmup_steps : int, optional
        Number of warmup steps for learning rate scheduler, by default 50
    lr_scheduler_type : str, optional
        Type of learning rate scheduler, by default "linear"
    gradient_accumulation_steps : int, optional
        Number of gradient accumulation steps, by default 64
    bf16 : bool, optional
        Whether to use bf16 precision, by default True
    num_train_epochs : int, optional
        Number of training epochs, by default 3
    weight_decay : float, optional
        Weight decay for optimizer, by default 0.01
    eval_strategy : str, optional
        Evaluation strategy, by default "steps"
    eval_steps : int, optional
        Number of steps between evaluations, by default 25
    save_strategy : str, optional
        Checkpoint save strategy, by default "steps"
    save_steps : int, optional
        Number of steps between saving checkpoints, by default 100
    logging_steps : int, optional
        Number of steps between logging, by default 100
    remove_unused_columns : bool, optional
        Whether to remove unused columns from the dataset, by default False

    Examples
    --------
    >>> python main.py train --data_dir=$DATA_DIR --output_dir=$OUTPUT_DIR --max_steps=500 --use_wandb=True
    """
    logger.info(f"Loading base model from {model_name}")
    base_model = load_base_model(model_name, task_type)

    logger.info(f"Configuring LoRA adapter")
    model = create_peft_model(base_model)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %d", trainable_params)
    logger.info("All parameters: %d", all_params)
    logger.info("Percent trainable: %.2f%%", trainable_params / all_params * 100)

    # Configure dataloaders and splits
    logger.info("Loading datasets")
    train_dataset = Dataset.from_parquet(
        str(Path(train_dir)),
        keep_in_memory=False,
    )
    eval_dataset = Dataset.from_parquet(
        str(Path(valid_dir)),
        keep_in_memory=False,
    )

    if crop_length > 0:
        logger.info(f"Limiting sequences to {crop_length} tokens")
        def crop_middle(example):
            input_ids = example["input_ids"]
            seq_len = len(input_ids)
            if seq_len < crop_length:
                raise ValueError(f"Sequence too short: {seq_len} < {crop_length}")
            start = (seq_len - crop_length) // 2
            end = start + crop_length
            example["input_ids"] = input_ids[start:end]
            return example
        
        train_dataset = train_dataset.map(crop_middle, desc=f"Cropping to middle {crop_length}")
        eval_dataset = eval_dataset.map(crop_middle, desc=f"Cropping to middle {crop_length}")


    logger.info(f"Train dataset:\n{train_dataset}")
    logger.info(f"Eval dataset:\n{eval_dataset}")

    if eval_num_samples:
        logger.info(f"Limiting eval dataset to {eval_num_samples} samples")
        eval_dataset = eval_dataset.select(
            range(min(eval_num_samples, len(eval_dataset)))
        )

    # Run training
    logger.info("Running training")

    if use_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project
    

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        bf16=bf16,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to="none" if not use_wandb else "wandb",
        run_name=wandb_run_name,
        remove_unused_columns=remove_unused_columns,
        save_total_limit=5
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_classification if task_type == "classification" else compute_metrics_regression,
    )
    if resume_from_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

def evaluate(
    checkpoint_dir: str,
    data_dir: str,
    output_dir: str = "/tmp/pcv2-ft-eval",
    model_name: str = None,
    task_type: str = "classification",
    batch_size: int = 32,
    sampling_rate: Optional[float] = None,
    seed: int = 42,
    crop_length: int = 0,
) -> None:
    """Evaluate a fine-tuned model checkpoint on validation data.

    Loads a trained LoRA adapter and evaluates performance on validation data.
    Computes metrics including accuracy, F1 score, ROC AUC, and average precision.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing the trained LoRA adapter checkpoint
    data_dir : str
        Directory containing tokenized validation data (valid.parquet)
    output_dir : str, optional
        Directory to save evaluation results
    model_name : str, optional
        Name or path of base model used during training
    batch_size : int, optional
        Batch size for evaluation, by default 32 (~28GB GPU memory)
    sampling_rate : float, optional
        If provided, evaluate on this fraction of validation data
    seed : int, optional
        Random seed for reproducibility, by default 42

    Examples
    --------
    >>> # Evaluate on 10% of validation data
    >>> python main.py evaluate --data_dir=$DATA_DIR --checkpoint_dir=$CHECKPOINT_DIR --sampling_rate=.1
    >>> # Evaluate on full validation set
    >>> python main.py evaluate --data_dir=$DATA_DIR --checkpoint_dir=$CHECKPOINT_DIR
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_dir}")
    base_model = load_base_model(model_name, task_type)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    # Load the dataset to evaluate
    logger.info(f"Loading dataset from {data_dir}")
    eval_dataset = Dataset.from_parquet(
        data_dir,
        split="test",
        keep_in_memory=False,
    )
    logger.info(f"Evaluation dataset:\n{eval_dataset}")

    if crop_length > 0:
        logger.info(f"Cropping evaluation sequences to {crop_length} tokens")
        def crop_middle(example):
            input_ids = example["input_ids"]
            seq_len = len(input_ids)
            if seq_len < crop_length:
                raise ValueError(f"Sequence too short: {seq_len} < {crop_length}")
            start = (seq_len - crop_length) // 2
            end = start + crop_length
            example["input_ids"] = input_ids[start:end]
            return example
        
        eval_dataset = eval_dataset.map(crop_middle, desc=f"Cropping to middle {crop_length}")

    if sampling_rate:
        if sampling_rate > 1 or sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be a float in (0, 1]; got {sampling_rate}")
        max_index = min(int(sampling_rate * len(eval_dataset)), len(eval_dataset))
        max_index = max(max_index, 1)
        eval_dataset = (
            eval_dataset.shuffle(seed=seed)
            .select(range(max_index))
        )
        logger.info(f"Evaluation dataset after sampling:\n{eval_dataset}")
    

    trainer_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
        compute_metrics=compute_metrics_classification if task_type == "classification" else compute_metrics_regression,
    )

    logger.info("Evaluating model on test split...")
    results = trainer.evaluate(eval_dataset=eval_dataset)
    logger.info("Evaluation complete")
    logger.info(f"Results:\n{results}")

def predict(
    checkpoint_dir: str,
    data_dir: str,
    output_file: str = "/tmp/predictions.csv",
    model_name: str = None,
    task_type: str = "classification",
    batch_size: int = 32,
    sampling_rate: Optional[float] = None,
    seed: int = 42,
    crop_length: int = 0,
) -> None:
    """Generate predictions for a dataset and save them to a CSV file.

    Loads a trained LoRA adapter and generates prediction scores for the dataset.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing the trained LoRA adapter checkpoint
    data_dir : str
        Directory containing tokenized data (valid.parquet)
    output_file : str, optional
        Path to save the predictions as a CSV file
    model_name : str, optional
        Name or path of base model used during training
    batch_size : int, optional
        Batch size for prediction, by default 32
    sampling_rate : float, optional
        If provided, predict on this fraction of the dataset
    seed : int, optional
        Random seed for reproducibility, by default 42

    Examples
    --------
    >>> # Predict on full dataset
    >>> python main.py predict --data_dir=$DATA_DIR --checkpoint_dir=$CHECKPOINT_DIR --output_file=predictions.csv
    >>> # Predict on 10% of the dataset
    >>> python main.py predict --data_dir=$DATA_DIR --checkpoint_dir=$CHECKPOINT_DIR --sampling_rate=0.1
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_dir}")
    base_model = load_base_model(model_name, task_type)
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    # Load the dataset to predict
    logger.info(f"Loading dataset from {data_dir}")
    dataset = Dataset.from_parquet(
        data_dir,
        split="test",
        keep_in_memory=False,
    )
    logger.info(f"Dataset:\n{dataset}")

    if crop_length > 0:
        logger.info(f"Cropping prediction sequences to {crop_length} tokens")
        def crop_middle(example):
            input_ids = example["input_ids"]
            seq_len = len(input_ids)
            if seq_len < crop_length:
                raise ValueError(f"Sequence too short: {seq_len} < {crop_length}")
            start = (seq_len - crop_length) // 2
            end = start + crop_length
            example["input_ids"] = input_ids[start:end]
            return example
        dataset = dataset.map(crop_middle, desc=f"Cropping to middle {crop_length}")

    if sampling_rate:
        if sampling_rate > 1 or sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be a float in (0, 1]; got {sampling_rate}")
        max_index = min(int(sampling_rate * len(dataset)), len(dataset))
        max_index = max(max_index, 1)
        dataset = (
            dataset.shuffle(seed=seed)
            .select(range(max_index))
        )
        logger.info(f"Dataset after sampling:\n{dataset}")

    trainer_args = TrainingArguments(
        output_dir="/tmp",  # Temporary directory for Trainer
        per_device_eval_batch_size=batch_size,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=trainer_args,
    )

    logger.info("Generating predictions...")
    predictions = trainer.predict(test_dataset=dataset).predictions

    if task_type == "classification":
        probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1).numpy()
        scores = probs[:, 1]  # Extract the scores for the positive class
        df = pd.DataFrame({"probability_positive": scores})
    else:  # regression
        # For regression, predictions are direct values
        values = predictions.squeeze()
        df = pd.DataFrame({"predicted_value": values})
    
    df.to_csv(output_file, index=False)
    logger.info("Prediction scores saved successfully")

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def compute_metrics_classification(eval_pred):
    predictions, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1)
    predictions = np.argmax(predictions, axis=1)
    scores = probs[:, 1].numpy()

    balance = np.sum(labels) / len(labels)
    f1 = f1_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    average_precision = average_precision_score(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'average_precision': average_precision,
        'balance': balance,
    }

def compute_metrics_regression(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    # Common regression metrics
    mse = ((predictions - labels) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(predictions - labels).mean()
    
    # Coefficient of determination (R²)
    # R² = 1 - (unexplained variance / total variance)
    ss_tot = ((labels - labels.mean()) ** 2).sum()
    ss_res = ((labels - predictions) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero

    pearson_corr, _ = pearsonr(predictions, labels)
    spearman_corr, _ = spearmanr(predictions, labels)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_corr,
        'spearman_r': spearman_corr,
    }


def load_base_model(model_name: str, task_type: str = "classification") -> AutoModelForSequenceClassification:
    if task_type == "classification":
        logger.info(f"Loading base model for classification from {model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            num_labels=2,
            id2label = {0: "NEGATIVE", 1: "POSITIVE"},
            label2id = {"NEGATIVE": 0, "POSITIVE": 1},
        )
    else:  # regression
        logger.info(f"Loading base model for regression from {model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            num_labels=1,  # Single output for regression
            problem_type="regression",
        )

    # Handle unsupported arguments in base_model
    original_forward = base_model.forward
    def forward_with_logging(*args, **kwargs):
        return original_forward(
            input_ids=kwargs['input_ids'], 
            labels=kwargs['labels']
        )
    base_model.forward = forward_with_logging

    return base_model

def create_peft_model(base_model: AutoModelForSequenceClassification) -> PeftModelForSequenceClassification:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["x_proj", "in_proj", "out_proj"],
    )
    return get_peft_model(base_model, peft_config)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({
        "tokenize": tokenize,
        "train": train,
        "evaluate": evaluate,
        "predict": predict,
        "display": display,
    })
