"""Cortex-MoE chat demo (local, CPU-friendly)."""

import argparse
import base64
import io
import os
import random

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def _preprocess_image(image, image_size=224):
    image = image.convert("RGB").resize((image_size, image_size))
    pixels = torch.tensor(image.getdata(), dtype=torch.float32) / 255.0
    pixels = pixels.view(image_size, image_size, 3).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (pixels - mean) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("CORTEX_MODEL_DIR", "cortex_moe_565m"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"loading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print("model loaded. type a message (or paste an image path) and press Enter.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        pixel_values = None
        if os.path.isfile(user_input):
            pixel_values = _preprocess_image(Image.open(user_input)).unsqueeze(0)
            user_input = "Describe this image."

        messages = [{"role": "user", "content": user_input}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        with torch.no_grad():
            out = model.generate(
                inputs,
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        print(f"Cortex-MoE: {reply}")


if __name__ == "__main__":
    main()
