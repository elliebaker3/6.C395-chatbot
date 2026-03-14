# judges/accuracy_judge.py
"""
Accuracy Judge

Evaluates whether the chatbot's response contains factually correct
information compared to a known ground truth answer.

Uses the same local Llama model as the chatbot itself.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

# module-level model load — shared across all judge calls
# avoids reloading the model on every judge invocation
_tokenizer = None
_model = None

def _get_model():
    """Lazy-loads the model once and reuses it."""
    global _tokenizer, _model
    if _model is None:
        print("[accuracy_judge] loading model...")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        _model.eval()
    return _tokenizer, _model


SYSTEM_PROMPT = """You are an expert evaluator assessing the factual accuracy of an AI chatbot 
that answers questions about the MIT course catalog.

You will be given:
- A question asked by a student
- The ground truth correct answer
- The chatbot's actual response

Your job is to determine whether the chatbot's response is factually accurate 
compared to the ground truth. Be strict about factual errors — wrong course 
numbers, incorrect prerequisites, or hallucinated courses should be penalized heavily.

You must respond in EXACTLY this format with nothing else:
SCORE: <integer 1-5>
REASON: <one sentence explanation>

Scoring rubric:
5 - Fully accurate, all facts match ground truth
4 - Mostly accurate, minor omissions but nothing wrong
3 - Partially accurate, some correct facts but notable errors or gaps
2 - Mostly inaccurate, significant factual errors
1 - Completely wrong or hallucinated answer"""


def judge_accuracy(question: str, response: str, ground_truth: str) -> dict:
    """
    Judges whether the chatbot response is factually accurate
    compared to the known ground truth.

    Args:
        question:     The original question asked
        response:     The chatbot's response to evaluate
        ground_truth: The known correct answer for this question

    Returns:
        dict with keys: criterion, score (int 1-5), reason (str)
    """
    tokenizer, model = _get_model()

    # format as Llama instruct chat template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Question: {question}

Ground truth answer: {ground_truth}

Chatbot response: {response}"""}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,   # score + one sentence fits easily
            temperature=0.1,      # low temp for consistent structured output
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # decode only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return _parse_response(raw)


def _parse_response(text: str) -> dict:
    """
    Parses the judge's SCORE/REASON response into a dict.
    Falls back gracefully if the format is unexpected.
    """
    try:
        lines = text.strip().split("\n")
        score_line  = next(l for l in lines if l.startswith("SCORE:"))
        reason_line = next(l for l in lines if l.startswith("REASON:"))

        score  = int(score_line.split(":")[1].strip())
        reason = reason_line.split(":", 1)[1].strip()

        # clamp to valid range
        score = max(1, min(5, score))

        return {"criterion": "accuracy", "score": score, "reason": reason}

    except Exception as e:
        print(f"[accuracy_judge] parse error: {e} | raw: {text}")
        return {
            "criterion": "accuracy",
            "score": 1,
            "reason": "Failed to parse judge response"
        }
