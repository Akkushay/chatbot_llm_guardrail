.. Chatbot LLM Guardrail documentation master file, created by sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root toctree directive.

Chatbot LLM Guardrail Documentation
===================================

Overview
--------
The Chatbot LLM Guardrail project is a modular system designed to evaluate the groundedness and consistency of banking chatbot responses. Its hybrid evaluation approach combines a fine-tuned LoRa-based classifier with a few-shot LLM prompt to ensure robust analysis. The project is organized into the following components:

- **Data Module:** For synthetic data generation.
- **Model Module:** For training and fine-tuning (using LoRa).
- **Evaluation Module:** For performing guardrail evaluations on responses.
- **App Module:** A Streamlit interface for interactive evaluation.

This documentation includes installation instructions, usage guidelines, API references, and design rationale.

Contents
--------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   modules
   design_decisions

Indices and Tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
