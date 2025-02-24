# DESIGN_DECISIONS.md

## Overview
This document outlines the key design decisions made in the LLM Guardrail Evaluation Project.

## 1. Use of LoRa for Fine-Tuning
- **Reasoning:** 
  - **Efficiency:** LoRa (Low-Rank Adaptation) reduces the number of trainable parameters, enabling efficient fine-tuning on limited resources.
  - **Preservation:** It adds lightweight adapters without overwriting the base model’s weights, preserving pretrained knowledge.
  - **Resource Constraints:** Ideal for our binary classification task (grounded vs. ungrounded) with faster training and lower GPU memory usage.

## 2. Hybrid Guardrail Approach
- **Components:**
  - A fast, LoRa-based classifier for initial scoring.
  - A detailed few-shot LLM prompt evaluation.
- **Reasoning:**
  - **Robust Evaluation:** Combining both ensures detection of unsupported claims and nuanced errors.
  - **Prompt Engineering:** The few-shot prompt—with clear examples and strict JSON instructions—ensures consistent and parseable output.
  - **Confidence Calibration:** Merging outputs from both models provides a more robust final decision.

## 3. Prompt Engineering Optimization
- **Structure:** 
  - Detailed examples and clear instructions for the LLM.
  - A focus on a strict JSON output to ease downstream processing.
- **Testing:** 
  - Iterative refinement and fallback mechanisms (e.g., regex fixes) were implemented to handle minor inconsistencies.

## 4. Testing Strategy
- **Unit Tests:** 
  - Comprehensive tests cover data preprocessing, model training, evaluation logic, and prompt parsing.
  - External dependencies (e.g., API calls) are mocked to ensure reliable tests.
- **Continuous Integration:** 
  - It is recommended to set up CI pipelines to run these tests automatically.

## Conclusion
This design optimizes for efficiency, clarity, and robustness. The project is modular and well tested, enabling easy extension and maintenance.
