"""Evaluate Cortex-MoE before / after fine-tuning on simple questions.

Runs the same prompt list through a base model and a fine-tuned model
and prints both outputs side by side, so you can judge the difference
by eye (and with a simple repetition / length metric).

Usage:
  python eval_before_after.py --base Frankenstein-Labs/Cortex-MoE-565M \\
      --ft cortex_moe_sft --questions eval_questions.jsonl
"""

import argparse
import json
import statistics

import torch
import cortex_moe  # noqa: registers "cortex_moe" in Auto* classes
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_questions(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def generate(model, tokenizer, prompt, max_new_tokens=32, do_sample=False, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def metrics(text):
    words = text.split()
    return {
        "tokens": len(words),
        "unique_ratio": round(len(set(words)) / max(1, len(words)), 3),
        "repeated_3gram": text.count(text[:12]) > 1 if len(text) > 12 else False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Frankenstein-Labs/Cortex-MoE-565M")
    parser.add_argument("--ft", default="cortex_moe_sft")
    parser.add_argument("--questions", default="eval_questions.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--ft-only", action="store_true")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.pad_token = tokenizer.eos_token

    models = {"base": AutoModelForCausalLM.from_pretrained(args.base)}
    if not args.ft_only:
        models["ft"] = AutoModelForCausalLM.from_pretrained(args.ft)
    for m in models.values():
        m.eval()

    print(f"{'Question':<60} | {'Base':<50} | {'FT':<50}")
    print("-" * 170)
    for q in questions:
        prompt = q["question"] if isinstance(q, dict) else q
        base_out = generate(models["base"], tokenizer, prompt, args.max_new_tokens)
        ft_out = generate(models["ft"], tokenizer, prompt, args.max_new_tokens) if "ft" in models else ""
        print(f"{prompt[:58]:<60} | {base_out[:48]:<50} | {ft_out[:48]:<50}")

    # aggregate metrics
    base_m = [metrics(generate(models["base"], tokenizer, q["question"] if isinstance(q, dict) else q, args.max_new_tokens)) for q in questions]
    print("\nBase avg tokens:", round(statistics.mean(m["tokens"] for m in base_m), 1),
          "| unique ratio:", round(statistics.mean(m["unique_ratio"] for m in base_m), 3))
    if "ft" in models:
        ft_m = [metrics(generate(models["ft"], tokenizer, q["question"] if isinstance(q, dict) else q, args.max_new_tokens)) for q in questions]
        print("FT   avg tokens:", round(statistics.mean(m["tokens"] for m in ft_m), 1),
              "| unique ratio:", round(statistics.mean(m["unique_ratio"] for m in ft_m), 3))


if __name__ == "__main__":
    main()