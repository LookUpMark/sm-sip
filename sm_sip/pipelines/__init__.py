"""End-to-end pipelines for training, inference, evaluation, and ablation."""

from sm_sip.pipelines.training import train_sigext, run_training_matrix
from sm_sip.pipelines.inference import run_inference, run_inference_pipeline
from sm_sip.pipelines.evaluation import run_evaluation, run_enhanced_evaluation
from sm_sip.pipelines.evaluation_sigext import run_sigext_evaluation_matrix
from sm_sip.pipelines.ablation import run_ablation_experiment
