# src/chatbot_llm_guardrail/evaluation/guardrail.py
import os
import re
import json
import math
import logging
import datetime
import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from chatbot_llm_guardrail.config import ADAPTER_DIR, GROQ_API_ENDPOINT, GROQ_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# def fix_supporting_docs(json_str: str) -> str:
#     """
#     Fixes the "supporting_docs" array so that all items are properly quoted.
#     """
#     pattern = r'"supporting_docs":\s*\[([^\]]+)\]'
#     def repl(match):
#         content = match.group(1).strip()
#         if not content:
#             return '"supporting_docs": []'
#         items = [item.strip() for item in content.split(",")]
#         fixed_items = [f'"{item}"' for item in items]
#         return '"supporting_docs": [' + ", ".join(fixed_items) + ']'
#     return re.sub(pattern, repl, json_str)

def fix_supporting_docs(json_str: str) -> str:
    """
    Fixes the "supporting_docs" array so that all items are properly quoted.
    """
    pattern = r'"supporting_docs":\s*\[([^\]]+)\]'
    def repl(match):
        content = match.group(1).strip()
        if not content:
            return '"supporting_docs": []'
        items = [item.strip() for item in content.split(",")]
        fixed_items = []
        for item in items:
            # Remove any surrounding quotes and wrap the item in double quotes
            item = item.strip('"')
            fixed_items.append(f'"{item}"')
        return '"supporting_docs": [' + ", ".join(fixed_items) + ']'
    return re.sub(pattern, repl, json_str)


class LLMGuardrail:
    """
    Provides advanced guardrail evaluation for chatbot responses using a hybrid approach:
    - A LoRa-based classifier loaded from a local adapter checkpoint.
    - A few-shot LLM prompt for additional evaluation.
    """

    def __init__(self):
        self.api_endpoint = GROQ_API_ENDPOINT
        self.api_key = GROQ_API_KEY
        if not self.api_endpoint or not self.api_key:
            raise EnvironmentError("GROQ_API_ENDPOINT and GROQ_API_KEY must be set as environment variables.")
        if not os.path.exists(ADAPTER_DIR):
            raise Exception("Adapter checkpoint not found. Please run the training script first.")
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=2
            )
            self.classifier_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.classifier_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            if hasattr(self.classifier_model, "set_active_adapters"):
                self.classifier_model.set_active_adapters("bank_adapter")
            self.classifier_model.eval()
        except Exception as e:
            raise Exception(f"Failed to load adapter-based classifier: {e}")

    # def _extract_json(self, text: str) -> dict:
    #     """
    #     Extracts and parses a JSON object from the given text.
    #     """
    #     match = re.search(r"(\{.*\})", text, re.DOTALL)
    #     if not match:
    #         logging.error("No JSON object found in LLM response: %s", text)
    #         raise Exception("No JSON object found in the response text.")
    #     json_str = match.group(1)
    #     json_str = fix_supporting_docs(json_str)
    #     try:
    #         return json.loads(json_str)
    #     except Exception as e:
    #         logging.error("JSON decoding failed for LLM response: %s", text)
    #         raise Exception(f"JSON decoding failed: {e}")

    def _extract_json(self, text: str) -> dict:
        """
        Extracts and parses a JSON object from the given text by capturing text from the first '{' to the last '}'.
        It then fixes the "supporting_docs" array if necessary.
        """
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            logging.error("No JSON object found in LLM response: %s", text)
            raise Exception("No JSON object found in the response text.")
        json_str = text[start:end+1]
        # Fix the supporting_docs array
        json_str = fix_supporting_docs(json_str)
        try:
            return json.loads(json_str)
        except Exception as e:
            logging.error("JSON decoding failed for LLM response: %s", text)
            raise Exception(f"JSON decoding failed: {e}")


    def _call_llm(self, prompt: str, model: str) -> dict:
        """
        Calls the LLM API with the provided prompt.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Respond ONLY with valid JSON (all strings must be in double quotes) and no extra text."},
                {"role": "user", "content": prompt}
            ]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            full_response = response.json()
            try:
                content_str = full_response["choices"][0]["message"]["content"]
                return self._extract_json(content_str)
            except Exception as e:
                logging.error("LLM raw response: %s", full_response)
                raise Exception(f"Failed to parse API response content: {e}")
        else:
            raise Exception(f"API call failed with status code {response.status_code}: {response.text}")

    def _classifier_inference(self, response: str, retrieved_docs: list) -> dict:
        """
        Performs inference using the adapter-based classifier.
        """
        input_text = response + " " + " ".join([doc.get("content", "") for doc in retrieved_docs])
        inputs = self.classifier_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = self.classifier_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            probability = probs[0][1].item()
        return {"probability": probability}

    def build_groundedness_prompt(self, response: str, retrieved_documents: list) -> str:
        """
        Builds a prompt for LLaMA-8B-8192 to evaluate whether `response` is grounded in `retrieved_documents`.
        """
        docs_text = ""
        for doc in retrieved_documents:
            docs_text += f"Document {doc.get('doc_id','Missing Document ID')}:\n{doc.get('content', 'No Document Found')}\n\n"
        prompt = f"""
                    You are a judge AI working for CommBank tasked with evaluating the groundedness of a chatbot's answer using provided documents.

                    Data:
                    - response (string): 
                    {response}

                    - retrieved_documents (list of strings): 
                    {docs_text}

                    Task: Determine if the `response` is fully supported by the `retrieved_documents`. Base your judgment only on the documents. Check each claim in the response:
                    - If all claims are supported, the response is grounded.
                    - If any claim is unsupported or contradicts the documents, it is not grounded.

                    Evaluate:
                    1. Support: Identify supporting document(s) for each claim.
                    2. Consistency: Verify no contradictions.
                    3. Completeness: Ensure all details in the response are found in the documents.
                    4. Competitor Reference: Ensure no mention of competitors (e.g., Westpac, ANZ, NAB).
                    5. Toxicity: Ensure no toxic language is used.

                    Scoring:
                    - groundedness_score: 1 if fully supported, 0 otherwise.
                    - consistency_score (0.0–1.0): Higher is better.
                    - consistency_label: "Low", "Medium", or "High".
                    - analysis: Brief explanation.
                    - supporting_docs: List of document IDs that support the claims.
                    - confidence: A float between 0.0 and 1.0.

                    Output Format: Provide a single JSON object with keys:

                    "groundedness_score","consistency_score","consistency_label","analysis","supporting_docs","confidence"

                    Now, analyze the response using the retrieved_documents and output the JSON.
        """
        return prompt.strip()

    def evaluate_response_few_shot(self, response: str, retrieved_docs: list) -> dict:
        classifier_result = self._classifier_inference(response, retrieved_docs)
        prompt = self.build_groundedness_prompt(response, retrieved_docs)
        try:
            llm_result = self._call_llm(prompt, model="llama3-8b-8192")
        except Exception as e:
            llm_result = {
                "groundedness_score": 0,
                "consistency_score": 0.0,
                "consistency_label": "Low",
                "analysis": f"LLM evaluation failed: {str(e)}",
                "supporting_docs": [],
                "confidence": 0.0
            }
        # Ensure numeric values
        llm_groundedness = llm_result.get("groundedness_score", 0)
        try:
            llm_groundedness = float(llm_groundedness)
        except Exception:
            llm_groundedness = 0.0
        llm_confidence = llm_result.get("confidence", 0.0)
        try:
            llm_confidence = float(llm_confidence)
        except Exception:
            llm_confidence = 0.0

        classifier_weight = 0
        confidence_weight = 0
        llm_weight = 1
        calibrated_score = (classifier_weight * classifier_result["probability"] +
                            llm_confidence * confidence_weight +
                            llm_groundedness * llm_weight)
        final_groundedness = 1 if calibrated_score >= 0.6 else 0
        llm_result["groundedness_score"] = final_groundedness
        llm_result["confidence"] = calibrated_score

        analysis = llm_result.get("analysis")
        if not analysis or not isinstance(analysis, str) or analysis.strip() == "":
            default_analysis = ("No relevant supporting documents found."
                                if final_groundedness == 0
                                else "Response is well-supported by the retrieved documents.")
            analysis = default_analysis
        llm_result["analysis"] = analysis

        supporting_docs = llm_result.get("supporting_docs")
        if not isinstance(supporting_docs, list):
            supporting_docs = [str(supporting_docs)]
        else:
            supporting_docs = [str(doc) for doc in supporting_docs]
        llm_result["supporting_docs"] = supporting_docs

        audit_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "response": response,
            "retrieved_docs": retrieved_docs,
            "classifier_result": classifier_result,
            "llm_result": llm_result
        }
        logging.info("Guardrail evaluation record: %s", json.dumps(audit_record))
        return llm_result

