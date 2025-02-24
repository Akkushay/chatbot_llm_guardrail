# LLM Guardrail Evaluation Project

## Overview
This project provides a hybrid evaluation system for banking chatbot responses. It consists of:
- **Synthetic data generator** for creating labeled examples.
- **LoRa-based classifier** fine-tuned on a banking dataset.
- **Hybrid guardrail system** that uses a large language model (LLM) prompt to evaluate response groundedness and consistency.
- **Streamlit app** for interactive evaluation.
- Comprehensive documentation generated via Sphinx.
- **Design decisions** [Design Decisions](DESIGN_DECISIONS.md) file  with discussion around why a particular design was used 

```plaintext
chatbot_llm_guardrail/
├── README.md
├── requirements.txt
├── DESIGN_DECISIONS.md
├── src/
│   └── chatbot_llm_guardrail/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── synthetic_data_builder.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── training.py
│       │   └── custom_trainer.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── guardrail.py
│       │   └── evaluator.py
│       └── app/
│           ├── __init__.py
│           └── streamlit_app.py
├── tests/
│   ├── __init__.py
│   ├── test_guardrail.py
│   ├── test_synthetic_data.py
│   ├── test_training.py
│   └── test_evaluator.py
└── docs/
    ├── conf.py
    ├── index.rst
    └── modules.rst
```



## Setup Instructions

1. **Clone the Repository:**

	git clone https://github.com/Akkushay/chatbot_llm_guardrail.git
	cd chatbot_llm_guardrail

2. **Create a Virtual Environment and Install Dependencies:**	
	python -m venv venv
	source venv/bin/activate  
	pip install -r requirements.txt

3. **Environment Variables:** Create a .env file in the root directory (this file is not tracked in version control) with the following variables:
	GROQ_API_ENDPOINT=<your_groq_api_endpoint>
	GROQ_API_KEY=<your_groq_api_key>
	OPENAI_API_KEY=<your_openai_api_key>

4. **Running the Training Script:** Fine-tune the LoRa-based classifier by running:
	python -m chatbot_llm_guardrail.model.training


5. **Launching the Streamlit App:**
	streamlit run src/chatbot_llm_guardrail/app/streamlit_app.py


6. **Generating Documentation with Sphinx:** Navigate to the docs/ directory and run:
	make html 
	```open _build/html/index.html``` to run html file on your browser

7. **Running Unit Tests: Run the tests using pytest:**
	pytest tests/


## Requirements
See requirements.txt for the complete list of dependencies.

## License
Copyright (c) [2025] [Akshay Sharma]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Additional Information
For detailed design decisions, refer to DESIGN_DECISIONS.md.
