Installation
============

Follow these steps to install the Chatbot LLM Guardrail project:

1. **Clone the Repository:**

   Open your terminal and run:
   ::
   
       git clone <repository_url>
       cd chatbot_llm_guardrail

2. **Create and Activate a Virtual Environment:**

   For Unix/Mac:
   ::
   
       python3 -m venv venv
       source venv/bin/activate
   
   For Windows:
   ::
   
       python -m venv venv
       venv\Scripts\activate

3. **Install Dependencies:**

   Run the following command to install the required packages:
   ::
   
       pip install -r requirements.txt

4. **Set Up Environment Variables:**

   Create a file named ``.env`` in the project root (next to README.md) with the following content:
   ::
   
       GROQ_API_ENDPOINT=<your_groq_api_endpoint>
       GROQ_API_KEY=<your_groq_api_key>
       OPENAI_API_KEY=<your_openai_api_key>
   
   *Note: Do not commit the .env file to version control.*

5. **Data Files:**

   Ensure that the data files ``bank_dataset_big.txt`` and ``bank_dataset.txt`` are placed in the expected location (by default, in the project root). Alternatively, update the file paths in ``src/chatbot_llm_guardrail/config.py`` as needed.

6. **(Optional) Build the Documentation:**

   To generate HTML documentation, navigate to the ``docs`` folder and run:
   ::
   
       sphinx-build -b html . _build/html
   
   Then open ``_build/html/index.html`` in your web browser.
