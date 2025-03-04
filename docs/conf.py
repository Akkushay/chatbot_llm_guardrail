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
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_baseurl = "https://akkushay.github.io/chatbot_llm_guardrail/"
html_use_relative_urls = True
