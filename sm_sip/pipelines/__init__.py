"""End-to-end pipelines for training, inference, and evaluation."""

from sm_sip.pipelines.training import train_sigext
from sm_sip.pipelines.inference import run_inference
from sm_sip.pipelines.evaluation import run_evaluation, run_enhanced_evaluation
