import os
from dotenv import load_dotenv

# Load from .env file. Store your HF token in the .env file.
load_dotenv()


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# Other options:
# BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"  # Alternative that might work better
# BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # Another alternative
# BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# If you finetune the model or change it in any way, save it to huggingface hub, then set MY_MODEL to your model ID. The model ID is in the format "your-username/your-model-name".
MY_MODEL = None

HF_TOKEN = os.getenv("HF_TOKEN")
