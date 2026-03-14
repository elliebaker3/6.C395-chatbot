import os
import sys
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# walk up from eval/ to project root so src.chat is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chat import Chatbot

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MY_MODEL   = None
HF_TOKEN   = os.getenv("HF_TOKEN")

client  = InferenceClient(model=MY_MODEL or BASE_MODEL, token=HF_TOKEN)
chatbot = Chatbot()

# set this to whichever judge you want to run
judge_type = "info_retrieval"

VALID_JUDGE_TYPES = [
    "info_retrieval",
    "multi_step_info_retrieval",
    "recommender",
    "off_topic",
    "followup_questions",
    "no_hallucination",
    "error_catcher",
    "bias_catcher",
]

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
TESTS_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "unit_tests.txt")


def load_unit_tests(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def load_system_prompt(judge_type: str) -> str:
    if judge_type not in VALID_JUDGE_TYPES:
        raise ValueError(f"Unknown judge_type '{judge_type}'. Must be one of: {VALID_JUDGE_TYPES}")
    path = os.path.join(PROMPTS_DIR, f"{judge_type}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path) as f:
        return f.read().strip()


def judge(question: str, ground_truth: str) -> dict:
    response = chatbot.get_response(question, history=[])

    system_prompt = load_system_prompt(judge_type)

    result = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""Does this response correctly answer the question given the ground truth? Also provide a confidence score about how sure you are of your score. This should be between 0 and 1.

Question: {question}
Ground truth: {ground_truth}
Response: {response}

Reply in this exact format:
SCORE: <1-5>
CONFIDENCE SCORE: <0-1>
REASON: <one sentence>"""

            }
        ],
        max_tokens=100,
        temperature=0.1
    )

    raw = result.choices[0].message.content.strip()
    lines = raw.split("\n")
    score_line  = next(l for l in lines if l.startswith("SCORE:"))
    confidence_score_line = next(l for l in lines if l.startswith("CONFIDENCE SCORE:"))
    reason_line = next(l for l in lines if l.startswith("REASON:"))

    return {
        "question":     question,
        "response":     response,
        "ground_truth": ground_truth,
        "score":        int(score_line.split(":")[1].strip()),
        "confidence_score":        float(confidence_score_line.split(":")[1].strip()),
        "reason":       reason_line.split(":", 1)[1].strip()
    }


# --- run all unit tests ---
unit_tests = load_unit_tests(TESTS_PATH)
print(f"Loaded {len(unit_tests)} test cases\n")

for question, ground_truth in unit_tests:
    result = judge(question, ground_truth)
    print(f"Question: {result['question']}")
    print(f"Response: {result['response']}")
    print(f"Score:    {result['score']}/5")
    print(f"Confidence Score:     {result['confidence_score']}")
    print(f"Reason:   {result['reason']}")
    print("-" * 60)
