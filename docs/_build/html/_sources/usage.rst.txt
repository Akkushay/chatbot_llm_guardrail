Usage
=====

This section describes how to run various components of the Chatbot LLM Guardrail project.

**Training the Model**
------------------------
To fine-tune the LoRa-based classifier, run the training script from the project root:
::
  
   python -m chatbot_llm_guardrail.model.training

The training script will:
- Load the dataset from ``bank_dataset_big.txt``.
- Preprocess and oversample the data.
- Fine-tune a DistilBERT model using LoRa.
- Save the adapter checkpoint to the ``adapter_checkpoint/`` directory.
- Generate diagnostic outputs (plots, CSVs) in the ``training_outputs/`` directory.

**Running the Streamlit App**
-----------------------------
To launch the interactive evaluation interface, run:
::
  
   streamlit run src/chatbot_llm_guardrail/app/streamlit_app.py

This app lets you:
- Evaluate individual chatbot responses.
- Perform bulk evaluations using the synthetic dataset (``bank_dataset.txt``).

**Bulk Evaluation**
--------------------
For bulk evaluation, ensure that ``bank_dataset.txt`` exists in the expected location or update the path in the configuration. The bulk evaluation runs within the Streamlit app and displays evaluation metrics and results.
