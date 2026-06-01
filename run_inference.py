import os
import re
import sys
import csv
import json
import torch
import gc
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"  
MAX_TOKENS  = 32768

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["VLLM_USE_V1"] = "0"

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician.\n"
    "Solve the problem quickly and give the correct answer.\n\n"
    
    "RULES:\n"
    "- Use the minimum number of steps required to reach the answer.\n"
    "- Do NOT explain concepts or restate the problem.\n"
    "- Do NOT second-guess yourself. Keep validating to a minimum.\n"
    "- Maintain absolute symbolic precision throughout intermediate work. Use only exact forms (fractions, radicals, or pi) until the final step.\n"
    "- Do NOT convert to decimals mid-calculation; this prevents rounding drift.\n"
    "- Stop immediately once the answer is found.\n\n"
    
    "- If the final answer is a decimal, round it to AT MOST 12 decimal places.\n"
    "- Do NOT include trailing zeros (e.g., write 5.67, not 5.670000000000).\n"
    "- If the answer is a whole number or a simple terminating decimal, provide it in its shortest form.\n"

    "OUTPUT FORMAT:\n"
    "- Final answer must be inside \\boxed{}.\n"
    "- Only apply decimal conversion/rounding in the final \\boxed{} answer.\n\n"
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, \n"
    "e.g. \\boxed{3, 7}.\n"
    "The sub-answers inside \boxed{} MUST EXACTLY follow the chronological order in which they were requested in the question.\n"
    "For any final answers left in formula/variable format, you MUST explicitly display all multiplications using the asterisk * symbol (e.g., write 5*x instead of 5x, and t*(t-1) instead of t(t-1)).\n"
    "EXAMPLE: \n"
    "Input: Calculate the hypotenuse and area of a right triangle with legs 3.1 and 4.2.\n"
    "<thought>\n"
    "Convert decimals to exact fractions to prevent mid-calculation drift: 3.1 = 31/10, 4.2 = 42/10.\n"
    "c^2 = (31/10)^2 + (42/10)^2 = 961/100 + 1764/100 = 2725/100 = 109/4.\n"
    "c = \\sqrt{109/4} = \\sqrt{109}/2.\n"
    "Problem uses decimals, final answer as decimal: \\sqrt{109} \\approx 10.44030650891.\n"
    "c \\approx 5.220153254455. Shortest terminating or max 12 decimal places, no trailing zeros.\n"
    "</thought>\n"
    "\\boxed{5.220153254455}\n"
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician.\n"
    "Solve the problem quickly and choose the correct option.\n\n"

    "RULES:\n"
    "- Do NOT use decimals during calculation unless the options are provided in decimal form.\n"
    "- Use the minimal computation needed to identify the answer among the choices.\n"
    "- If no option is mathematically perfect, select the choice that is analytically closest or intended.\n"
    "- Stop as soon as the correct choice is identified.\n\n"

    "OUTPUT FORMAT:\n"
    "- Output ONLY the final capital letter of the correct option inside \\boxed{} (e.g., \\boxed{A}).\n"
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, \n"
    "e.g. \\boxed{C, D}.\n"
)

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_boxed_answer(text: str) -> str:
    """Extracts content from \\boxed{}. Handles up to one level of nested braces."""
    matches = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', text)
    return matches[-1].strip() if matches else text.strip()


def run_inference(input_path: str = "data/private.jsonl", output_path: str = "submission.csv"):
    """
    End-to-end pipeline: Loads model, runs inference, applies post-processing,
    and outputs the final submission CSV.
    """
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
    else:
        print("WARNING: No GPU available.")

    print(f"Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    print("Initializing Tokenizer and vLLM Engine...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.95,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=8,
        max_num_batched_tokens=32768,
        dtype="bfloat16"
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        top_p=0.9,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.3,
    )

    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    responses = [out.outputs[0].text.strip() for out in outputs]

    for i in range(min(3, len(responses))):
        print(f"\n── Response {i} (id={data[i].get('id')}) ──")
        print(responses[i][:400], "..." if len(responses[i]) > 400 else "")

    results = []
    for idx, (item, response) in tqdm(enumerate(zip(data, responses)), total=len(prompts), desc="Scoring"):
        results.append({
            "id":            item.get("id"),
            "response":      response,
        })

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving final submission to {out_path}...")
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "response"]) 
        for r in results:
            writer.writerow([r["id"], r["response"]])

    print("Pipeline complete. CSV generated successfully.")

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_inference(input_path="data/private.jsonl", output_path="submission.csv")