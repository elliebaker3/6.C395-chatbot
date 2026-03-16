import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so it works regardless of current working directory
_project_root = Path(__file__).resolve().parent
load_dotenv(_project_root / ".env")


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# Other options:
# BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"  # Alternative that might work better
# BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # Another alternative
# BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# If you finetune the model or change it in any way, save it to huggingface hub, then set MY_MODEL to your model ID. The model ID is in the format "your-username/your-model-name".
MY_MODEL = None

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    import warnings
    warnings.warn(
        "HF_TOKEN is not set. Set it in .env (e.g. HF_TOKEN=hf_xxx) or run: huggingface-cli login",
        UserWarning,
        stacklevel=1,
    )
