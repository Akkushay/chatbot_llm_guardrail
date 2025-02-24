# src/chatbot_llm_guardrail/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Endpoints and Keys
GROQ_API_ENDPOINT = os.environ.get("GROQ_API_ENDPOINT")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# File paths and directories
DATA_FILE = "data/bank_dataset_big.txt"
INFERENCE_DATA_FILE = "data/bank_dataset.txt"
CHECKPOINT_DIR = "adapter_checkpoint"
ADAPTER_DIR = os.path.join(CHECKPOINT_DIR, "bank_adapter")
TRAINING_OUTPUTS_DIR = "training_outputs"
