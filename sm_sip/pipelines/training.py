"""
SigExt model training pipeline.

Handles the full training loop: semantic label generation,
tokenization, training with AdamW, and optional push to HuggingFace Hub.

Optimized for batch training: datasets, similarities, and tokenizations
are cached and reused across configs that share the same language or
base model, avoiding redundant computation.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from sm_sip.config import TrainingConfig
from sm_sip.data.dataset import SigExtDataset
from sm_sip.data.preprocessing import (
    compute_similarities,
    extract_sentences,
    labels_from_similarities,
)
from sm_sip.utils.gpu import clear_gpu_memory


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _get_dataset_keys(entry: dict) -> Tuple[str, str]:
    """Heuristic to find source and summary keys."""
    keys = entry.keys()
    source_key = next((k for k in ["article", "source", "text", "document"] if k in keys), "source")
    summary_key = next((k for k in ["abstract", "summary", "target", "label"] if k in keys), "summary")
    return source_key, summary_key


def _load_raw_dataset(dataset_name: str, num_samples: int) -> List[Dict[str, str]]:
    """Load raw entries from HuggingFace and normalize keys to source/summary."""
    from datasets import load_dataset as hf_load_dataset

    print(f"  Loading dataset: {dataset_name} ({num_samples} samples)...")
    ds = hf_load_dataset(dataset_name, split="train", streaming=True)

    entries = []
    source_key = summary_key = None
    for entry in ds:
        if source_key is None:
            source_key, summary_key = _get_dataset_keys(entry)
        entries.append({"source": entry[source_key], "summary": entry[summary_key]})
        if len(entries) >= num_samples:
            break

    print(f"  Loaded {len(entries)} entries.")
    return entries


# ---------------------------------------------------------------------------
# Similarity pre-computation
# ---------------------------------------------------------------------------

def _precompute_similarities(
    entries: List[Dict[str, str]],
    lang: str,
) -> List[Tuple[List[str], np.ndarray, str]]:
    """Pre-compute SBERT similarities for all entries (once per language group).

    Returns list of (sentences, similarities, source_text) tuples.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Pre-computing similarities ({len(entries)} samples, lang={lang})...")
    sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    results = []
    for entry in tqdm(entries, desc="  Computing similarities"):
        sents, sims = compute_similarities(
            entry["source"], entry["summary"], lang=lang, sbert_model=sbert,
        )
        results.append((sents, sims, entry["source"]))

    del sbert
    clear_gpu_memory()
    return results


# ---------------------------------------------------------------------------
# Token-level label alignment
# ---------------------------------------------------------------------------

def _build_dataset(
    sim_data: List[Tuple[List[str], np.ndarray, str]],
    threshold: float,
    tokenizer,
    max_length: int,
) -> SigExtDataset:
    """Build a SigExtDataset from pre-computed similarities.

    Applies the threshold to derive labels and aligns them to tokens.
    """
    all_texts = []
    all_labels = []

    for sentences, similarities, source in sim_data:
        if len(similarities) == 0:
            continue

        labels = labels_from_similarities(similarities, threshold)

        encoding = tokenizer(
            source, truncation=True, max_length=max_length,
            return_offsets_mapping=True,
        )

        token_labels = [0] * len(encoding["input_ids"])
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
        all_texts, truncation=True, max_length=max_length,
        padding=True, return_tensors="pt",
    )
    return SigExtDataset(encodings, all_labels)


# ---------------------------------------------------------------------------
# Single model training
# ---------------------------------------------------------------------------

def _train_single(
    config: TrainingConfig,
    dataset: SigExtDataset,
) -> None:
    """Train a single SigExt model on a pre-built dataset."""
    from sm_sip.utils.seed import set_seed

    set_seed(config.seed)

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {config.output_model_name} (seed={config.seed})")
    print(f"  Base: {config.base_model_id}")
    print(f"  Threshold: {config.similarity_threshold}, Samples: {len(dataset)}")
    print(f"{'=' * 60}")

    g = torch.Generator()
    g.manual_seed(config.seed)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, generator=g,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model_id, num_labels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01,
    )

    # Mixed precision (FP16) for faster training and lower VRAM usage
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"  Epoch {epoch + 1}/{config.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1} loss: {total_loss / len(dataloader):.4f}")

    # Save / Push to hub
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, use_fast=False)
    if config.push_to_hub:
        print(f"  Pushing to Hub: {config.output_model_name}...")
        model.push_to_hub(config.output_model_name)
        tokenizer.push_to_hub(config.output_model_name)
    else:
        save_path = f"./models/{config.output_model_name}"
        print(f"  Saving to: {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    del model, optimizer, scaler
    clear_gpu_memory()
    print("  Done.")


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str) -> List[str]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_checkpoint(path: str, completed: List[str]) -> None:
    with open(path, "w") as f:
        json.dump(completed, f, indent=2)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_training_matrix(
    configs: List[TrainingConfig],
    checkpoint_file: str = "training_checkpoint.json",
) -> None:
    """Run the full training matrix with optimal caching.

    Optimizations:
      - Datasets are loaded once per (lang, dataset_name).
      - SBERT similarities are computed once per language group.
      - Tokenization is done once per (base_model_id, sample_subset).
      - Checkpointing allows safe resumption.

    Args:
        configs: List of TrainingConfig for each model to train.
        checkpoint_file: Path to checkpoint JSON.
    """
    completed = _load_checkpoint(checkpoint_file)
    remaining = [c for c in configs if c.output_model_name not in completed]

    print(f"Total configs: {len(configs)}, already completed: {len(completed)}, "
          f"remaining: {len(remaining)}")

    if not remaining:
        print("All models already trained!")
        return

    # --- Group by (lang, dataset_name) for dataset + similarity caching ---
    lang_groups: Dict[str, List[TrainingConfig]] = defaultdict(list)
    for c in remaining:
        lang_groups[(c.lang, c.dataset_name)].append(c)

    for (lang, dataset_name), group_configs in lang_groups.items():
        print(f"\n{'#' * 70}")
        print(f"# Language group: {lang.upper()} — {dataset_name}")
        print(f"# Models to train: {len(group_configs)}")
        print(f"{'#' * 70}")

        # 1. Load dataset once (max samples needed in this group)
        max_samples = max(c.num_samples for c in group_configs)
        raw_entries = _load_raw_dataset(dataset_name, max_samples)

        # 2. Pre-compute similarities once
        sim_data = _precompute_similarities(raw_entries, lang)

        # 3. Group by base_model_id for tokenizer reuse
        model_groups: Dict[str, List[TrainingConfig]] = defaultdict(list)
        for c in group_configs:
            model_groups[c.base_model_id].append(c)

        for base_model_id, model_configs in model_groups.items():
            print(f"\n  Tokenizer: {base_model_id}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)

            for config in model_configs:
                if config.output_model_name in completed:
                    print(f"  Skip {config.output_model_name} (already completed)")
                    continue

                # Slice similarity data to num_samples
                sliced_sim = sim_data[:config.num_samples]

                # Build dataset (threshold + tokenize — fast, no SBERT)
                dataset = _build_dataset(
                    sliced_sim, config.similarity_threshold,
                    tokenizer, config.max_length,
                )

                try:
                    _train_single(config, dataset)
                    completed.append(config.output_model_name)
                    _save_checkpoint(checkpoint_file, completed)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    clear_gpu_memory()
                    continue

            del tokenizer

        del raw_entries, sim_data
        clear_gpu_memory()

    print(f"\n\nAll {len(configs)} training runs complete!")


# ---------------------------------------------------------------------------
# Backward-compatible single-model entry point
# ---------------------------------------------------------------------------

def train_sigext(config: TrainingConfig, dataset_entries: Optional[list] = None):
    """Train a single SigExt model (backward-compatible entry point).

    For batch training of multiple models, use run_training_matrix() instead.
    """
    run_training_matrix([config])
