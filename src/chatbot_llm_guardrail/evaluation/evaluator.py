# src/chatbot_llm_guardrail/evaluation/evaluator.py
import os
import json
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from chatbot_llm_guardrail.evaluation.guardrail import LLMGuardrail
from chatbot_llm_guardrail.config import INFERENCE_DATA_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Evaluate:
    """
    Provides methods to evaluate chatbot responses, either one at a time or in bulk.
    """

    def __init__(self):
        self.guardrail = LLMGuardrail()
        self.dataset_file = INFERENCE_DATA_FILE

    def evaluate_one_sample(self, response: str, retrieved_docs: list) -> dict:
        """
        Evaluate a single sample using the guardrail.
        """
        return self.guardrail.evaluate_response_few_shot(response, retrieved_docs)

    def evaluate_all_samples(self):
        """
        Evaluate all samples in the synthetic dataset.
        Returns:
            evaluation_df: DataFrame with columns: Query, Response, Grounded, Output.
            metrics_dict: Dictionary with overall metrics.
        """
        samples = []
        if not os.path.exists(self.dataset_file):
            raise Exception(f"Dataset file {self.dataset_file} not found.")
        with open(self.dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        records = []
        true_labels = []
        predicted_labels = []
        for sample in samples:
            query = sample.get("query", "")
            response = sample.get("response", "")
            retrieved_docs = sample.get("retrieved_docs", [])
            synthetic_groundedness = sample.get("synthetic_groundedness", 0)
            eval_result = self.guardrail.evaluate_response_few_shot(response, retrieved_docs)
            predicted_groundedness = eval_result.get("groundedness_score", 0)
            grounded = "Yes" if predicted_groundedness >= 0.5 else "No"
            output = "Correct" if predicted_groundedness == synthetic_groundedness else "Misclassified"
            records.append({
                "Query": query,
                "Response": response,
                "Grounded": grounded,
                "Output": output
            })
            true_labels.append(synthetic_groundedness)
            predicted_labels.append(1 if predicted_groundedness >= 0.5 else 0)
        evaluation_df = pd.DataFrame(records)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        return evaluation_df, metrics_dict
