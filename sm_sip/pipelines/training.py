"""
SigExt model training pipeline.

Handles the full training loop: semantic label generation,
tokenization, training with AdamW, and optional push to HuggingFace Hub.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from sm_sip.config import TrainingConfig
from sm_sip.data.dataset import SigExtDataset
from sm_sip.data.preprocessing import compute_semantic_labels, extract_sentences
from sm_sip.utils.gpu import clear_gpu_memory


def get_dataset_keys(entry: dict):
    """Heuristic to find source and summary keys."""
    keys = entry.keys()
    source_key = next((k for k in ["article", "source", "text", "document"] if k in keys), "source")
    summary_key = next((k for k in ["abstract", "summary", "target", "label"] if k in keys), "summary")
    return source_key, summary_key


def prepare_training_data(
    config: TrainingConfig,
    dataset_entries: list,
) -> SigExtDataset:
    """Generate semantically labeled training data.

    For each sample, compute per-sentence salience labels using
    Sentence-BERT similarity, then align labels to tokens.

    Args:
        config: Training configuration.
        dataset_entries: List of dicts with 'source' and 'summary' keys.

    Returns:
        SigExtDataset ready for DataLoader.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=False)
    lang = config.lang

    all_texts = []
    all_labels = []

    # Find keys from first entry
    if dataset_entries:
        source_key, summary_key = get_dataset_keys(dataset_entries[0])
    else:
        source_key, summary_key = "source", "summary"

    for entry in tqdm(dataset_entries[:config.num_samples], desc="Preparing training data"):
        source = entry[source_key]
        summary = entry[summary_key]

        sentences, labels = compute_semantic_labels(
            source, summary,
            threshold=config.similarity_threshold,
            lang=lang,
        )

        if not sentences:
            continue

        # Tokenize and align labels to tokens
        encoding = tokenizer(
            source,
            truncation=True,
            max_length=config.max_length,
            return_offsets_mapping=True,
        )

        # Simple alignment: assign label based on sentence boundaries
        token_labels = [0] * len(encoding["input_ids"])
        # Mark tokens corresponding to salient sentences
        for sent, label in zip(sentences, labels):
            if label == 1:
                start_idx = source.find(sent)
                if start_idx >= 0:
                    end_idx = start_idx + len(sent)
                    for j, (s, e) in enumerate(encoding["offset_mapping"]):
                        if s >= start_idx and e <= end_idx and s != e:
                            token_labels[j] = 1

        all_texts.append(source)
        all_labels.append(token_labels)

    encodings = tokenizer(
        all_texts,
        truncation=True,
        max_length=config.max_length,
        padding=True,
        return_tensors="pt",
    )

    return SigExtDataset(encodings, all_labels)


def train_sigext(config: TrainingConfig, dataset_entries: Optional[list] = None):
    """Full training pipeline for SigExt model.

    Args:
        config: Training configuration.
        dataset_entries: Optional pre-loaded dataset entries.
                        If None, will load from HuggingFace Datasets.
    """
    from datasets import load_dataset as hf_load_dataset
    from sm_sip.utils.seed import set_seed

    # Set seed for reproducibility
    set_seed(config.seed)

    print("=" * 60)
    print(f"TRAINING: {config.output_model_name} (seed={config.seed})")
    print("=" * 60)

    # Load data if not provided
    if dataset_entries is None:
        print(f"  Loading dataset: {config.dataset_name}...")
        ds = hf_load_dataset(config.dataset_name, split="train", streaming=True)
        dataset_entries = []
        for entry in ds:
            dataset_entries.append(entry)
            if len(dataset_entries) >= config.num_samples:
                break

    # Prepare training data
    print("  Preparing training data...")
    train_dataset = prepare_training_data(config, dataset_entries)

    # Seeded DataLoader for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(config.seed)
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=g)

    # Load model
    print(f"  Loading base model: {config.base_model_id}...")
    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model_id, num_labels=2
    ).to("cuda")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"  Epoch {epoch + 1}/{config.epochs}"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1} loss: {total_loss / len(dataloader):.4f}")

    # Save / Push to hub
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=False)
    if config.push_to_hub:
        print(f"  Pushing to HuggingFace Hub: {config.output_model_name}...")
        model.push_to_hub(config.output_model_name)
        tokenizer.push_to_hub(config.output_model_name)
    else:
        save_path = f"./models/{config.output_model_name}"
        print(f"  Saving to: {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    clear_gpu_memory()
    print("  Training complete!")
    return model, tokenizer
