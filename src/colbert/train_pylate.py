#!/usr/bin/env python3
"""
ColBERT Training Script using PyLate
====================================

This script fine-tunes a ColBERT model using the PyLate library with contrastive learning.
"""

import json
import logging
import os
import random
import torch
import wandb
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_pylate_format(data, max_negatives_per_query=1, random_seed=42):
    """
    Convert data from the original format to pylate/sentence-transformers format.
    
    Args:
        data (list): List of dictionaries with keys:
            ['query', 'positive_id', 'positive_code', 'negative_ids', 'negative_codes', 
             'repo', 'base_commit', 'instance_id', 'negative_code_scores', 
             'negative_code_rank', 'positive_code_score', 'positive_code_rank']
        max_negatives_per_query (int): Maximum number of negative examples to use per query.
            If a query has more negatives, they will be randomly sampled.
        random_seed (int): Random seed for reproducible negative sampling.
    
    Returns:
        Dataset: HuggingFace Dataset in pylate format with train/test split.
    """
    logger.info(f"Converting {len(data)} examples to pylate format")
    
    # Set random seed for reproducible sampling
    random.seed(random_seed)
    
    converted_data = []
    
    for item in data:
        query = item['query']
        positive = item['positive_code']
        negative_codes = item['negative_codes']
        
        # Handle case where negative_codes might be a single string or list
        if isinstance(negative_codes, str):
            negatives = [negative_codes]
        else:
            negatives = negative_codes
        
        # Sample negatives if there are too many
        if len(negatives) > max_negatives_per_query:
            negatives = random.sample(negatives, max_negatives_per_query)
        
        # Create one training example for each negative
        for negative in negatives:
            converted_data.append({
                "query": query,
                "positive": positive,
                "negative": negative,
            })
    
    logger.info(f"Created {len(converted_data)} training examples after conversion")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(converted_data)
    
    return dataset


def convert_jsonl_to_pylate_format(file_path, max_negatives_per_query=1, test_size=0.3, random_seed=42):
    """
    Load JSONL file and convert to pylate format with train/test split.
    
    Args:
        file_path (str): Path to the JSONL file.
        max_negatives_per_query (int): Maximum number of negative examples per query.
        test_size (float): Proportion of data to use for testing (0.0 to 1.0).
        random_seed (int): Random seed for reproducible splits.
    
    Returns:
        tuple: (train_dataset, test_dataset) as HuggingFace Datasets.
    """
    logger.info(f"Loading and converting dataset from: {file_path}")
    
    # Load raw data
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(data)} raw examples")
    
    # Convert to pylate format
    dataset = convert_to_pylate_format(data, max_negatives_per_query, random_seed)
    
    # Split into train and test
    if test_size > 0:
        splits = dataset.train_test_split(test_size=test_size, seed=random_seed)
        train_dataset = splits["train"]
        test_dataset = splits["test"]
        logger.info(f"Split dataset - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        return train_dataset, test_dataset
    else:
        logger.info(f"No test split - Train: {len(dataset)}")
        return dataset, None


def load_jsonl_dataset(file_path):
    """Load dataset from JSONL file."""
    logger.info(f"Loading dataset from: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(data)} examples from dataset")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    return dataset


def main():
    """Main training function."""
    # Configuration
    data_file = "/home/joe/SweRank/src/repo_contrastive_mined_filtered.jsonl"
    model_name = "lightonai/Reason-ModernColBERT"
    output_dir = "/home/joe/SweRank/src/colbert/finetuned_colbert_full"
    
    # Training parameters
    num_train_epochs = 2
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    learning_rate = 2e-5
    warmup_steps = 10
    eval_steps = 300
    save_steps = 300
    logging_steps = 5
    temperature = 0.02
    eval_dataset_size = 0.001
    max_negatives_per_query = 8  # Number of negatives to use per query
    
    # Initialize wandb
    wandb.init(
        project="colbert-finetuning",
        name="colbert-pylate-training",
        config={
            "model_name": model_name,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "temperature": temperature,
            "eval_dataset_size": eval_dataset_size,
            "max_negatives_per_query": max_negatives_per_query,
            "gradient_accumulation_steps": 4,
        },
        tags=["colbert", "pylate", "contrastive-learning"]
    )
    
    logger.info("Starting ColBERT training with PyLate")
    logger.info(f"Model: {model_name}")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Wandb run: {wandb.run.name}")
    
    # Load and convert dataset to pylate format
    train_dataset, eval_dataset = convert_jsonl_to_pylate_format(
        data_file, 
        max_negatives_per_query=max_negatives_per_query,
        test_size=eval_dataset_size,
        random_seed=42
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Log dataset info to wandb
    wandb.log({
        "dataset/train_size": len(train_dataset),
        "dataset/eval_size": len(eval_dataset),
        "dataset/total_size": len(train_dataset) + len(eval_dataset)
    })
    
    # Create model
    logger.info(f"Initializing ColBERT model from: {model_name}")
    model = models.ColBERT(model_name_or_path=model_name)
    
    # Note: Commenting out model compilation as it may cause compatibility issues
    logger.info("Compiling model for faster training")
    model = torch.compile(model)
    
    # Create loss function
    logger.info(f"Using Contrastive loss with temperature={temperature}")
    loss_fn = losses.Contrastive(
        model=model,
        temperature=temperature,
        gather_across_devices=False  # Enable for multi-GPU training
    )
    
    # Create evaluator
    logger.info("Creating ColBERTTripletEvaluator")
    evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        fp16=False,  # Re-enable mixed precision for faster training
        bf16=True,
        run_name="colbert-finetuning",
        logging_steps=logging_steps,
        eval_strategy="epoch",
        # eval_steps=eval_steps,
        save_strategy="epoch",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=False,  # Disabled due to torch.compile compatibility
        # metric_for_best_model="eval_accuracy",
        # greater_is_better=True,
        report_to=["wandb"],  # Enable wandb logging
        dataloader_drop_last=True,
        remove_unused_columns=False,
        gradient_accumulation_steps=32,  # Restore gradient accumulation
    )
    
    # Create data collator
    data_collator = utils.ColBERTCollator(tokenize_fn=model.tokenize)
    
    # Initialize trainer
    logger.info("Initializing SentenceTransformerTrainer")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss_fn,
        evaluator=evaluator,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    logger.info(f"Saving final model to: {final_model_path}")
    model.save_pretrained(final_model_path)
    
    # Log final model path to wandb
    wandb.log({"model/final_path": final_model_path})
    
    # Finish wandb run
    wandb.finish()
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
