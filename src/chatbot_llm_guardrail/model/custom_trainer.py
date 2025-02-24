# src/chatbot_llm_guardrail/model/custom_trainer.py
import torch
from transformers import Trainer

class CustomTrainer(Trainer):
    """
    A custom Trainer subclass that uses CrossEntropyLoss for classification.
    """
    def compute_loss(self, model, inputs, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        if kwargs.get("return_outputs", False):
            return loss, outputs
        return loss
