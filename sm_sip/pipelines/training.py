"""
SigExt model training pipeline.

Handles the full training loop: semantic label generation,
tokenization, training with AdamW, and optional push to HuggingFace Hub.

Optimized for batch training: datasets, similarities, and tokenizations
are cached and reused across configs that share the same language or
base model, avoiding redundant computation.
"""

import glob
import json
import os
import pickle
import re
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


def load_raw_dataset(dataset_name: str, num_samples: int) -> List[Dict[str, str]]:
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

def precompute_similarities_data(
    entries: List[Dict[str, str]],
    lang: str,
    cache_dir: str = "cache",
    save_every: int = 500,
) -> List[Tuple[List[str], np.ndarray, str]]:
    """Pre-compute SBERT similarities for all entries (once per language group).

    Results are persisted to disk so that a kernel restart does NOT
    require re-running SBERT.  Partial results are saved every
    `save_every` samples, allowing safe resumption from any point.

    Returns list of (sentences, similarities, source_text) tuples.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"similarities_{lang}_{len(entries)}.pkl")

    # --- Improved Cache Discovery ---
    pattern = os.path.join(cache_dir, f"similarities_{lang}_*.pkl")
    existing_files = glob.glob(pattern)
    
    best_path = None
    max_found = -1
    
    for path in existing_files:
        match = re.search(rf"similarities_{lang}_(\d+)\.pkl", os.path.basename(path))
        if match:
            count = int(match.group(1))
            if count > max_found:
                max_found = count
                best_path = path

    results: List[Tuple[List[str], np.ndarray, str]] = []
    
    # If the exact file exists, we'll use it. 
    # Otherwise, if we found a "best" alternative, we load it.
    load_path = cache_path if os.path.exists(cache_path) else best_path

    if load_path:
        print(f"  Loading existing cache: {os.path.basename(load_path)}")
        with open(load_path, "rb") as f:
            results = pickle.load(f)
            
        if len(results) >= len(entries):
            print(f"  Cache has {len(results)} samples (needed {len(entries)}). Truncating.")
            results = results[:len(entries)]
            # If we loaded from a different file, save it to the current target path
            if load_path != cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump(results, f)
            return results
        else:
            print(f"  Resuming from {len(results)}/{len(entries)} samples...")

    from sentence_transformers import SentenceTransformer

    print(f"  Pre-computing similarities ({len(entries) - len(results)} remaining, lang={lang})...")
    sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    for i, entry in enumerate(tqdm(entries[len(results):], desc="  Computing similarities",
                                    initial=len(results), total=len(entries))):
        sents, sims = compute_similarities(
            entry["source"], entry["summary"], lang=lang, sbert_model=sbert,
        )
        results.append((sents, sims, entry["source"]))

        # Incremental save
        if (len(results)) % save_every == 0:
            with open(cache_path, "wb") as f:
                pickle.dump(results, f)

    del sbert
    clear_gpu_memory()

    # Final save
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Similarities cached to {cache_path}")

    return results


# ---------------------------------------------------------------------------
# Token-level label alignment
# ---------------------------------------------------------------------------

def build_sigext_dataset(
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

    # Use the FAST tokenizer for offset_mapping (slow tokenizers don't support it)
    fast_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer.name_or_path, use_fast=True,
    )

    for sentences, similarities, source in sim_data:
        if len(similarities) == 0:
            continue

        labels = labels_from_similarities(similarities, threshold)

        encoding = fast_tokenizer(
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

    # Pad labels to match the padded sequence length
    # Using -100 so they are ignored by CrossEntropyLoss
    max_seq_len = encodings.input_ids.shape[1]
    padded_labels = []
    for labels in all_labels:
        pad_len = max_seq_len - len(labels)
        padded_labels.append(labels + [-100] * pad_len)

    return SigExtDataset(encodings, padded_labels)


# ---------------------------------------------------------------------------
# Single model training
# ---------------------------------------------------------------------------

def _train_single(
    config: TrainingConfig,
    train_dataset: SigExtDataset,
    val_dataset: SigExtDataset,
) -> None:
    """Train a single SigExt model with early stopping on validation loss."""
    import copy
    from sm_sip.utils.seed import set_seed

    set_seed(config.seed)

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {config.output_model_name} (seed={config.seed})")
    print(f"  Base: {config.base_model_id}")
    print(f"  Threshold: {config.similarity_threshold}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"  Early stopping: patience={config.patience}, max_epochs={config.epochs}")
    print(f"{'=' * 60}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model_id, num_labels=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01,
    )

    # --- Compute class weights (inverse frequency) ---
    counts = torch.zeros(2)
    for sample_labels in train_dataset.labels:
        for lbl in sample_labels:
            if lbl >= 0:  # skip -100 padding
                counts[lbl] += 1
    class_weights = (counts.sum() / (2 * counts)).to(device)
    print(f"  Class distribution: non-salient={int(counts[0])}, salient={int(counts[1])}")
    print(f"  Class weights: [{class_weights[0]:.2f}, {class_weights[1]:.2f}]")

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    # Mixed precision (FP16)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Early stopping state ---
    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(config.epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"  Epoch {epoch + 1}/{config.epochs} [train]")):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                logits = outputs.logits  # (batch, seq_len, 2)
                loss = loss_fn(logits.view(-1, 2), labels.view(-1))
                # Normalize loss for accumulation
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % config.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * config.gradient_accumulation_steps

        avg_train = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(**batch)
                    logits = outputs.logits
                    loss = loss_fn(logits.view(-1, 2), labels.view(-1))
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        improved = avg_val < best_val_loss

        print(f"  Epoch {epoch + 1}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}"
              f"  {'✓ improved' if improved else ''}")

        if improved:
            best_val_loss = avg_val
            # Store weights on CPU to avoid doubling GPU VRAM usage
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f})")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"  Restored best weights from epoch {best_epoch}")

    # Move model to CPU before saving to free GPU VRAM immediately
    model.cpu()
    clear_gpu_memory()

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

    del model, optimizer, scaler, best_weights
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

        # 1. Load dataset once (max samples needed + extra for validation)
        max_samples = max(c.num_samples for c in group_configs)
        max_val = max(int(c.num_samples * c.val_split) for c in group_configs)
        total_needed = max_samples + max_val
        raw_entries = load_raw_dataset(dataset_name, total_needed)

        # 2. Pre-compute similarities once (for all samples including val)
        sim_data = precompute_similarities_data(raw_entries, lang)

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

                # Train: first num_samples | Val: next val_size samples (no overlap)
                val_size = max(1, int(config.num_samples * config.val_split))
                train_sim = sim_data[:config.num_samples]
                val_sim = sim_data[config.num_samples:config.num_samples + val_size]

                train_dataset = build_sigext_dataset(
                    train_sim, config.similarity_threshold,
                    tokenizer, config.max_length,
                )
                val_dataset = build_sigext_dataset(
                    val_sim, config.similarity_threshold,
                    tokenizer, config.max_length,
                )

                try:
                    _train_single(config, train_dataset, val_dataset)
                    completed.append(config.output_model_name)
                    _save_checkpoint(checkpoint_file, completed)
                except Exception as e:
                    print(f"  FAILED: {e}")
                finally:
                    # Always clean up datasets and GPU between models
                    del train_dataset, val_dataset
                    clear_gpu_memory()

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
