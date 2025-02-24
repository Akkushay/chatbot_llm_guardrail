# src/chatbot_llm_guardrail/evaluation/evaluator.py
import os
import json
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import evaluate
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
        try:
            self.meteor_metric = evaluate.load("meteor")
            self.rouge_metric = evaluate.load("rouge")
            self.bertscore_metric = evaluate.load("bertscore")
        except Exception as e:
            logging.error("Failed to load evaluation metrics: %s", e)
            self.meteor_metric = None
            self.rouge_metric = None
            self.bertscore_metric = None

    def _compute_document_metrics(self, response: str, ref: str) -> dict:
        """
        Computes METEOR, ROUGE-L F1, and BERTScore for a single reference (ref)
        given the chatbot response.
        Returns a dict with keys: 'meteor', 'rougeL', 'bertscore'.
        """
        metrics = {"meteor": 0.0, "rougeL": 0.0, "bertscore": 0.0}
        if not ref:
            return metrics
        try:
            meteor_res = self.meteor_metric.compute(predictions=[response], references=[ref])
            metrics["meteor"] = float(meteor_res.get("meteor", 0.0))
        except Exception as e:
            logging.error(f"METEOR computation error: {e}")
        try:
            rouge_res = self.rouge_metric.compute(
                predictions=[response],
                references=[ref],
                rouge_types=["rougeL"]
            )
            if "rougeL" in rouge_res:
                metrics["rougeL"] = float(rouge_res["rougeL"])
            elif "rougeL_fmeasure" in rouge_res:
                metrics["rougeL"] = float(rouge_res["rougeL_fmeasure"])
        except Exception as e:
            logging.error(f"ROUGE-L computation error: {e}")
        try:
            bert_res = self.bertscore_metric.compute(
                predictions=[response],
                references=[ref],
                model_type="distilbert-base-uncased"
            )
            if bert_res.get("f1") and isinstance(bert_res["f1"], list) and len(bert_res["f1"]) > 0:
                metrics["bertscore"] = float(bert_res["f1"][0])
        except Exception as e:
            logging.error(f"BERTScore computation error: {e}")
        return metrics

    def evaluate_one_sample(self, response: str, retrieved_docs: list) -> dict:
        """
        Evaluate a single sample using the guardrail and compute additional metrics:
          - METEOR
          - ROUGE-L F1
          - BERTScore
        For each retrieved document, metrics are computed separately and then the maximum score is used.
        """
        eval_result = self.guardrail.evaluate_response_few_shot(response, retrieved_docs)
        doc_metrics = {"meteor": [], "rougeL": [], "bertscore": []}
        for doc in retrieved_docs:
            ref = doc.get("content", "").strip()
            if not ref:
                continue
            m = self._compute_document_metrics(response, ref)
            doc_metrics["meteor"].append(m["meteor"])
            doc_metrics["rougeL"].append(m["rougeL"])
            doc_metrics["bertscore"].append(m["bertscore"])
        max_meteor = max(doc_metrics["meteor"]) if doc_metrics["meteor"] else 0.0
        max_rouge = max(doc_metrics["rougeL"]) if doc_metrics["rougeL"] else 0.0
        max_bert = max(doc_metrics["bertscore"]) if doc_metrics["bertscore"] else 0.0
        eval_result["METEOR"] = max_meteor
        eval_result["ROUGE-L F1"] = max_rouge
        eval_result["BERTScore"] = max_bert
        return eval_result

    def evaluate_all_samples(self):
        """
        Evaluate all samples in the synthetic dataset.
        Returns:
            evaluation_df: DataFrame with columns: Query, Response, Grounded, Output.
            metrics_dict: Dictionary with overall classification metrics and average
                          METEOR, ROUGE-L F1, and BERTScore.
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
        meteor_list = []
        rouge_list = []
        bert_list = []
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
            # Compute per-sample metrics by evaluating each retrieved document and taking max
            sample_meteor = 0.0
            sample_rouge = 0.0
            sample_bert = 0.0
            doc_meteor = []
            doc_rouge = []
            doc_bert = []
            for doc in retrieved_docs:
                ref = doc.get("content", "").strip()
                if not ref:
                    continue
                m = self._compute_document_metrics(response, ref)
                doc_meteor.append(m["meteor"])
                doc_rouge.append(m["rougeL"])
                doc_bert.append(m["bertscore"])
            if doc_meteor:
                sample_meteor = max(doc_meteor)
            if doc_rouge:
                sample_rouge = max(doc_rouge)
            if doc_bert:
                sample_bert = max(doc_bert)
            meteor_list.append(sample_meteor)
            rouge_list.append(sample_rouge)
            bert_list.append(sample_bert)
        evaluation_df = pd.DataFrame(records)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        avg_meteor = float(sum(meteor_list) / len(meteor_list)) if meteor_list else 0.0
        avg_rouge = float(sum(rouge_list) / len(rouge_list)) if rouge_list else 0.0
        avg_bert = float(sum(bert_list) / len(bert_list)) if bert_list else 0.0
        metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "METEOR": avg_meteor,
            "ROUGE-L F1": avg_rouge,
            "BERTScore": avg_bert
        }
        return evaluation_df, metrics_dict
