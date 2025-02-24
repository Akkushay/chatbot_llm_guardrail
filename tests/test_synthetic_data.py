# tests/test_synthetic_data.py
import pytest
from chatbot_llm_guardrail.data.synthetic_data_builder import SyntheticDataBuilder

def test_generate_samples():
    builder = SyntheticDataBuilder(proportion_grounded=0.5, sample_count=3)
    samples = builder.generate_samples()
    assert len(samples) == 3
    for sample in samples:
        assert "query" in sample
        assert "response" in sample
        assert "retrieved_docs" in sample
        assert "synthetic_groundedness" in sample
        assert len(sample["retrieved_docs"]) == 5
