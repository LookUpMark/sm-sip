"""
Intrinsic SigExt evaluation pipeline.

Computes token-level classification metrics (Precision, Recall, F1)
for trained SigExt models against Sentence-BERT "gold" labels.
This allows ranking models based purely on their extraction
capabilities without involving the LLM generator.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader

from sm_sip.config import TrainingConfig
from sm_sip.pipelines.training import (
    load_raw_dataset,
    precompute_similarities_data,
    build_sigext_dataset,
)
from sm_sip.utils.gpu import clear_gpu_memory

def calculate_sentence_metrics(
    preds_seq: List[int],    # Flattened token predictions for a single document
    gold_labels: np.ndarray, # Sentence ground truth (1 if salient, 0 otherwise)
    offset_mapping: List[Tuple[int, int]], # Token offsets
    sentences: List[str],    # Original sentences
    source_text: str         # Full source text to locate sentences
) -> Dict[str, float]:
    """Compute P, R, F1 at the sentence level instead of token level.
    
    A sentence is considered 'salient' if any of its valid tokens are 
    predicted/labeled as salient.
    """
    sentence_pred = []
    
    # Pre-calculate token centers for faster matching
    token_centers = [(s + e) / 2 for s, e in offset_mapping]
    
    for sent in sentences:
        start_idx = source_text.find(sent)
        if start_idx == -1:
            sentence_pred.append(0)
            continue
            
        end_idx = start_idx + len(sent)
        total_tokens = 0
        pred_tokens = 0
        
        # Count total tokens and predicted tokens within this sentence
        for i, center in enumerate(token_centers):
            if start_idx <= center <= end_idx:
                total_tokens += 1
                if i < len(preds_seq) and preds_seq[i] == 1:
                    pred_tokens += 1
                    
        # Consider sentence extracted if > 50% of its tokens are predicted as salient
        sent_is_pred = total_tokens > 0 and (pred_tokens / total_tokens) > 0.5
        sentence_pred.append(1 if sent_is_pred else 0)
        
    # Now compare with gold_labels arrays
    gold_list = [1 if g == 1 else 0 for g in gold_labels]
    
    # Account for parsing issues if sentences > gold labels
    min_len = min(len(gold_list), len(sentence_pred))
    gold_list = gold_list[:min_len]
    sentence_pred = sentence_pred[:min_len]
    
    tp = sum(1 for g, p in zip(gold_list, sentence_pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold_list, sentence_pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold_list, sentence_pred) if g == 1 and p == 0)
    
    return {"tp": tp, "fp": fp, "fn": fn}

def evaluate_sigext_model(
    config: TrainingConfig,
    test_sim: List[Tuple[List[str], np.ndarray, str]],
    tokenizer,
    device: str = "cuda"
) -> Dict[str, float]:
    """Run intrinsic evaluation on a single model at the sentence level."""
    
    # Load model and tokenizer
    # Check if local model exists, else use HuggingFace
    model_path = f"./models/{config.output_model_name}"
    import os
    if not os.path.exists(model_path):
        model_name = config.output_model_name # Assume HF Hub
    else:
        model_name = model_path
        
    print(f"  Evaluating: {config.output_model_name}")
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        model.eval()
        
        # Use fast tokenizer to get offsets
        fast_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer.name_or_path, use_fast=True,
        )
        
        all_metrics = []
        
        with torch.no_grad():
            for sentences, similarities, source in tqdm(test_sim, desc="    Inference", leave=False):
                if not sentences:
                    continue
                
                # Derive gold labels
                from sm_sip.data.preprocessing import labels_from_similarities
                gold_labels = labels_from_similarities(similarities, config.similarity_threshold)
                
                # Tokenize
                encoding = fast_tokenizer(
                    source, truncation=True, max_length=config.max_length,
                    return_offsets_mapping=True, return_tensors="pt"
                )
                
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                offsets = encoding["offset_mapping"][0].tolist() # Remove batch dim
                
                # Predict
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)[0].tolist() # Remove batch dim
                
                metrics = calculate_sentence_metrics(
                    preds, gold_labels, offsets, sentences, source
                )
                all_metrics.append(metrics)
                
        # Aggregate
        tp = sum(m["tp"] for m in all_metrics)
        fp = sum(m["fp"] for m in all_metrics)
        fn = sum(m["fn"] for m in all_metrics)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        del model
        clear_gpu_memory()
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "total_salient_sentences": int(tp + fn)
        }
    except Exception as e:
        print(f"    Error evaluating {config.output_model_name}: {e}")
        return {}

def run_sigext_evaluation_matrix(
    configs: List[TrainingConfig],
    test_samples: int = 500,
    results_path: str = "results/sigext_intrinsic_metrics.json"
) -> Dict:
    """Run evaluation for all provided configs on a held-out test set."""
    import json
    import os
    
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Group by (lang, dataset_name, similarity_threshold) for dataset efficiency
    groups = {}
    for c in configs:
        key = (c.lang, c.dataset_name, c.similarity_threshold)
        if key not in groups:
            groups[key] = []
        groups[key].append(c)
        
    all_results = {}
    
    for (lang, dname, threshold), group_configs in groups.items():
        print(f"\nEvaluating Group: {lang} | {dname} | Threshold={threshold}")
        
        # Load test entries: Skip the MAX train + val pool
        max_train = max(c.num_samples for c in group_configs)
        max_val = max(int(c.num_samples * c.val_split) for c in group_configs)
        start_idx = max_train + max_val
        
        print(f"  Fetching test set starting from index {start_idx}...")
        test_entries = load_raw_dataset(dname, start_idx + test_samples)[start_idx:]
        
        # Only compute similarites for test_entries
        test_sim = precompute_similarities_data(test_entries, f"{lang}_test_{threshold}")
        
        # Group by base_model_id for tokenizer reuse
        sub_groups = {}
        for c in group_configs:
            if c.base_model_id not in sub_groups:
                sub_groups[c.base_model_id] = []
            sub_groups[c.base_model_id].append(c)
            
        for base_model_id, configs_in_sub in sub_groups.items():
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
            
            # Build test dataset once per base model (threshold is fixed in this outer loop)
            # test_dataset = build_sigext_dataset(
            #     test_sim, threshold, tokenizer, configs_in_sub[0].max_length
            # )
            
            for config in configs_in_sub:
                metrics = evaluate_sigext_model(config, test_sim, tokenizer, device)
                if metrics:
                    all_results[config.output_model_name] = metrics
                    
            del tokenizer
            
    # Save results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nIntrinsic evaluation complete. Results saved to {results_path}")
    return all_results
