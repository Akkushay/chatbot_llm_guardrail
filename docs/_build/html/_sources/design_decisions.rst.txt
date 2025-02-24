Design Decisions
================

This document explains the key design choices for the Chatbot LLM Guardrail project.

**Modular Architecture**
------------------------
The project is divided into distinct modules:
- **Data Module:** Contains synthetic data generation utilities.
- **Model Module:** Handles model training, data preprocessing, and custom training routines using LoRa.
- **Evaluation Module:** Implements the guardrail evaluation logic, combining classifier inference with a few-shot LLM prompt.
- **App Module:** Provides an interactive Streamlit interface for testing and evaluation.

**Why Use LoRa for Fine-Tuning?**
---------------------------------
- **Efficiency:** LoRa reduces the number of trainable parameters, making fine-tuning faster and less resource-intensive.
- **Preservation:** It retains most of the pretrained model's weights, ensuring that valuable pre-learned representations are maintained.
- **Scalability:** This approach is ideal for tasks with limited data, as it efficiently adapts the model for binary classification tasks (grounded vs. ungrounded).

**Hybrid Evaluation Approach**
------------------------------
- **Classifier Inference:** A lightweight LoRa-based classifier provides an initial, fast evaluation.
- **Few-shot LLM Prompt:** A structured prompt with clear examples is used to perform a detailed evaluation.
- **Calibration:** Outputs from both components are combined to produce a final evaluation score, ensuring robustness.

**Prompt Engineering**
----------------------
- **Structured Prompt:** The prompt includes explicit examples and instructions to generate valid JSON, reducing output errors.
- **Error Handling:** The code contains mechanisms (e.g., regex fixes) to correct common JSON formatting issues.
- **Iterative Refinement:** The prompt was tuned through testing to optimize clarity and performance.

**Testing Strategy**
--------------------
- **Unit Tests:** Comprehensive tests cover data preprocessing, training routines, evaluation logic, and API interactions.
- **Mocking:** External API calls are mocked during testing to ensure reliability and speed.
- **Continuous Integration:** The project is designed to integrate into CI pipelines to automatically run tests on each change.

For more details, please refer to the accompanying DESIGN_DECISIONS.md file in the project root.
