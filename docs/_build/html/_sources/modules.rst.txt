Project Modules
===============

This section provides an API reference for the key modules in the Chatbot LLM Guardrail project, along with a brief description of each module’s responsibilities.

Data Modules
------------
The Data Module is responsible for synthetic data generation. It includes functionality to generate synthetic samples for banking products, which are used for training and evaluation.

.. automodule:: chatbot_llm_guardrail.data.synthetic_data_builder
   :members:
   :undoc-members:
   :show-inheritance:
   :caption: Synthetic Data Builder Module

Model Modules
-------------
The Model Module contains the code for model training and preprocessing. It handles tasks such as loading datasets, oversampling to balance classes, tokenizing inputs, and fine-tuning the model using LoRa. It also includes a custom trainer class that extends the Hugging Face Trainer.

.. automodule:: chatbot_llm_guardrail.model.training
   :members:
   :undoc-members:
   :show-inheritance:
   :caption: Training and Preprocessing Module

.. automodule:: chatbot_llm_guardrail.model.custom_trainer
   :members:
   :undoc-members:
   :show-inheritance:
   :caption: Custom Trainer Module

Evaluation Modules
------------------
The Evaluation Module provides the core logic to assess the quality of chatbot responses. It features a guardrail that combines a LoRa-based classifier with a few-shot LLM prompt for detailed evaluation, and an evaluator class that can process single or bulk evaluations.

.. automodule:: chatbot_llm_guardrail.evaluation.guardrail
   :members:
   :undoc-members:
   :show-inheritance:
   :caption: Guardrail Evaluation Module

.. automodule:: chatbot_llm_guardrail.evaluation.evaluator
   :members:
   :undoc-members:
   :show-inheritance:
   :caption: Evaluator Module

App Modules
-----------
The App Module contains the Streamlit application which offers an interactive interface to test individual responses and perform bulk evaluations. It integrates the evaluation components with user-friendly visualizations.

.. automodule:: chatbot_llm_guardrail.app.streamlit_app
   :members:
   :undoc-members:
   :show-inheritance:
   :caption: Streamlit Application Module
