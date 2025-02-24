# tests/test_evaluator.py
import pytest
from chatbot_llm_guardrail.evaluation.evaluator import Evaluate

@pytest.mark.skip(reason="LLM API call - requires valid credentials and network access")
def test_evaluate_one_sample():
    evaluator = Evaluate()
    response = "Test response for evaluation."
    retrieved_docs = [{"doc_id": "DOC1", "content": "Test doc content."}]
    result = evaluator.evaluate_one_sample(response, retrieved_docs)
    assert "groundedness_score" in result
