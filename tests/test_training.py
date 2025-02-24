# tests/test_training.py
import os
import json
import tempfile
import pandas as pd
from chatbot_llm_guardrail.model.training import load_bank_dataset, preprocess_function, compute_metrics, delete_existing_checkpoint

def test_delete_existing_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "adapter_checkpoint"
    checkpoint_dir.mkdir()
    dummy_file = checkpoint_dir / "dummy.txt"
    dummy_file.write_text("dummy")
    original = os.environ.get("CHECKPOINT_DIR")
    os.environ["CHECKPOINT_DIR"] = str(checkpoint_dir)
    delete_existing_checkpoint()
    assert not checkpoint_dir.exists()
    if original:
        os.environ["CHECKPOINT_DIR"] = original

def test_preprocess_function():
    examples = {
        "response": ["Test response"],
        "retrieved_docs": [[{"content": "Doc content"}]],
        "synthetic_groundedness": [1]
    }
    tokenized = preprocess_function(examples)
    assert "input_ids" in tokenized
    assert "labels" in tokenized

def test_compute_metrics():
    import numpy as np
    logits = np.array([[1.0, 2.0]])
    labels = [1]
    metrics = compute_metrics((logits, labels))
    assert "accuracy" in metrics
