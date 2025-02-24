# src/chatbot_llm_guardrail/model/training.py
import os
import json
import logging
import shutil
import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
)

from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

from chatbot_llm_guardrail.config import DATA_FILE, INFERENCE_DATA_FILE, CHECKPOINT_DIR, ADAPTER_DIR, TRAINING_OUTPUTS_DIR
from chatbot_llm_guardrail.model.custom_trainer import CustomTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global tokenizer variable
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def delete_existing_checkpoint():
    """Deletes the existing checkpoint directory if it exists."""
    if os.path.exists(CHECKPOINT_DIR):
        logging.info("Deleting existing checkpoint directory...")
        shutil.rmtree(CHECKPOINT_DIR)
        logging.info("Existing checkpoint directory deleted.")

def load_bank_dataset():
    """
    Loads the bank dataset from DATA_FILE.
    Each line is a JSON record with fields: "response", "retrieved_docs", "synthetic_groundedness".
    """
    if not os.path.exists(DATA_FILE):
        logging.error(f"Dataset file '{DATA_FILE}' not found.")
        raise FileNotFoundError(f"Dataset file '{DATA_FILE}' not found.")

    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if "synthetic_groundedness" not in record:
                    raise KeyError("Dataset record missing 'synthetic_groundedness' field.")
                records.append(record)

    if not records:
        logging.error("No records found in the dataset.")
        raise ValueError("Dataset is empty.")

    return records

def preprocess_function(examples):
    """
    Concatenates 'response' and retrieved document contents, and assigns 'synthetic_groundedness' as labels.
    """
    inputs = []
    for response, docs in zip(examples["response"], examples["retrieved_docs"]):
        docs_text = " ".join([str(doc.get("content", "")) for doc in docs if doc is not None])
        inputs.append(response + " " + docs_text)

    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=256)
    if "synthetic_groundedness" not in examples:
        raise KeyError("Dataset missing 'synthetic_groundedness' field.")
    model_inputs["labels"] = examples["synthetic_groundedness"]
    return model_inputs

def compute_metrics(eval_pred):
    """
    Computes Accuracy, Precision, Recall, and F1 score.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }

def main():
    """Main training script."""
    delete_existing_checkpoint()

    try:
        raw_records = load_bank_dataset()
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        exit(1)

    full_dataset = Dataset.from_list(raw_records)
    split_dataset = full_dataset.train_test_split(test_size=0.1)
    train_dataset_hf = split_dataset["train"]
    test_dataset_hf = split_dataset["test"]

    df_train = pd.DataFrame(train_dataset_hf)
    X_train = df_train.drop(columns=["synthetic_groundedness"])
    y_train = df_train["synthetic_groundedness"]

    oversampler = RandomOverSampler()
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
    df_train_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    train_dataset_resampled = Dataset.from_pandas(df_train_resampled)

    train_tokenized = train_dataset_resampled.map(preprocess_function, batched=True)
    test_tokenized = test_dataset_hf.map(preprocess_function, batched=True)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logging.error("PEFT is not installed. Please install it using 'pip install peft'.")
        exit(1)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        logging_steps=50,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
        label_names=["labels"],
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    logging.info("Starting training with oversampled dataset...")
    trainer.train()

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    logging.info("Training complete. Adapter checkpoint saved.")

    eval_metrics = trainer.evaluate()
    logging.info(f"Evaluation Metrics: {eval_metrics}")

    base_output_dir = TRAINING_OUTPUTS_DIR
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Training outputs will be saved in: {output_dir}")

    TEMPERATURE = 1.0
    train_pred_output = trainer.predict(train_tokenized)
    train_logits = train_pred_output.predictions
    train_labels_np = np.array(train_pred_output.label_ids)
    train_probs = torch.softmax(torch.tensor(train_logits) / TEMPERATURE, dim=1).numpy()[:, 1]
    train_preds = np.argmax(train_logits, axis=1)

    val_pred_output = trainer.predict(test_tokenized)
    val_logits = val_pred_output.predictions
    val_labels_np = np.array(val_pred_output.label_ids)
    val_probs = torch.softmax(torch.tensor(val_logits) / TEMPERATURE, dim=1).numpy()[:, 1]
    val_preds = np.argmax(val_logits, axis=1)

    fpr_train, tpr_train, _ = roc_curve(train_labels_np, train_probs)
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_val, tpr_val, _ = roc_curve(val_labels_np, val_probs)
    roc_auc_val = auc(fpr_val, tpr_val)
    plt.figure()
    plt.plot(fpr_train, tpr_train, label=f'Training ROC (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_val, tpr_val, label=f'Validation ROC (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()

    precision_train, recall_train, _ = precision_recall_curve(train_labels_np, train_probs)
    auprc_train = average_precision_score(train_labels_np, train_probs)
    precision_val, recall_val, _ = precision_recall_curve(val_labels_np, val_probs)
    auprc_val = average_precision_score(val_labels_np, val_probs)
    plt.figure()
    plt.plot(recall_train, precision_train, label=f'Train (AUCPR = {auprc_train:.2f})')
    plt.plot(recall_val, precision_val, label=f'Val (AUCPR = {auprc_val:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    pr_curve_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_curve_path)
    plt.close()

    cm = confusion_matrix(val_labels_np, val_preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Validation)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Class 0", "Class 1"])
    plt.yticks(tick_marks, ["Class 0", "Class 1"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    prob_true, prob_pred = calibration_curve(val_labels_np, val_probs, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label="Validation")
    plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly Calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    calib_curve_path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(calib_curve_path)
    plt.close()

    log_history = trainer.state.log_history
    train_epochs, train_losses, eval_epochs, eval_losses = [], [], [], []
    for log in log_history:
        if "epoch" in log:
            if "loss" in log:
                train_epochs.append(log["epoch"])
                train_losses.append(log["loss"])
            if "eval_loss" in log:
                eval_epochs.append(log["epoch"])
                eval_losses.append(log["eval_loss"])
    if train_epochs and eval_epochs:
        plt.figure()
        plt.plot(train_epochs, train_losses, label="Training Loss")
        plt.plot(eval_epochs, eval_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curves")
        plt.legend()
        learning_curve_path = os.path.join(output_dir, "learning_curves.png")
        plt.savefig(learning_curve_path)
        plt.close()

    val_probs_reshaped = val_probs.reshape(-1, 1)
    calibration_model = LogisticRegression(solver='lbfgs')
    calibration_model.fit(val_probs_reshaped, val_labels_np)
    logging.info("Calibration model fitted using validation set predictions.")

    if not os.path.exists(INFERENCE_DATA_FILE):
        logging.error(f"Inference dataset file '{INFERENCE_DATA_FILE}' not found.")
    else:
        records_infer = []
        with open(INFERENCE_DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records_infer.append(json.loads(line))
        if records_infer:
            inference_dataset = Dataset.from_list(records_infer)
            inference_tokenized = inference_dataset.map(preprocess_function, batched=True)
            inference_pred = trainer.predict(inference_tokenized)
            logits_inference = inference_pred.predictions
            inference_probs = torch.softmax(torch.tensor(logits_inference) / TEMPERATURE, dim=1).numpy()[:, 1]
            inference_probs_reshaped = inference_probs.reshape(-1, 1)
            calibrated_scores = calibration_model.predict_proba(inference_probs_reshaped)[:, 1]
            csv_rows = []
            for record, raw_prob, calib_prob in zip(records_infer, inference_probs, calibrated_scores):
                response = record.get("response", "")
                retrieved_docs = record.get("retrieved_docs", [])
                retrieved_doc = " ".join([str(doc.get("content", "")) for doc in retrieved_docs if doc is not None])
                synthetic_groundedness = record.get("synthetic_groundedness", "")
                csv_rows.append({
                    "response": response,
                    "retrieved_doc": retrieved_doc,
                    "synthetic_groundedness": synthetic_groundedness,
                    "predicted score": raw_prob,
                    "calibrated score": calib_prob,
                })
            df_output = pd.DataFrame(csv_rows)
            csv_output_path = os.path.join(output_dir, "predicted_output_LoRa.csv")
            df_output.to_csv(csv_output_path, index=False)
            logging.info(f"Inference CSV saved to {csv_output_path}")
        else:
            logging.error("No records found in inference dataset.")

    logging.info(f"All training artifacts have been saved to {output_dir}.")

if __name__ == "__main__":
    main()
