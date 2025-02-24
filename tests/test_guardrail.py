# tests/test_guardrail.py
import pytest
from chatbot_llm_guardrail.evaluation.guardrail import fix_supporting_docs, LLMGuardrail

def test_fix_supporting_docs():
    input_json = '{"supporting_docs": [DOC123, DOC456]}'
    fixed = fix_supporting_docs(input_json)
    assert '"supporting_docs": ["DOC123", "DOC456"]' in fixed

@pytest.mark.skip(reason="LLM API call - requires valid credentials and network access")
def test_llm_guardrail_inference():
    guardrail = LLMGuardrail()
    response = "Test response."
    retrieved_docs = [{"doc_id": "DOC1", "content": "Test content."}]
    result = guardrail.evaluate_response_few_shot(response, retrieved_docs)
    assert "groundedness_score" in result
