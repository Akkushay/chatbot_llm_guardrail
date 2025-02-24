# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
project = 'Chatbot LLM Guardrail'
author = 'Akshay Sharma'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# Use the Read the Docs theme for a modern look.
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_baseurl = "https://github.com/Akkushay/chatbot_llm_guardrail.git/"
