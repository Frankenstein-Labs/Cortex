"""SFT (Supervised Fine-Tuning) of Cortex-MoE on instruction/response pairs.

Two modes:
  --lora   LoRA adapters only (CPU-friendly, ~1-4 GB RAM)
  --full   Full parameter training (needs GPU, ~12 GB VRAM for 565M)

Dataset: Hugging Face datasets, defaulting to a small French
instruction dataset (legmlai/openhermes-fr). Any instruction-style
dataset with `instruction`/`response` columns works.

Usage:
  python sft_cortex_moe.py --lora --dataset legmlai/openhermes-fr \\
      --output cortex_moe_sft_lora
"""

import argparse
import json
import os

import torch
import cortex_moe  # noqa: registers "cortex_moe" in Auto* classes
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_OK = True
except ImportError:
    PEFT_OK = False


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get("instruction") or item.get("question") or item.get("input")
        response = item.get("response") or item.get("output") or item.get("answer")
        text = f"### Instruction\n{prompt}\n### Réponse\n{response}{self.tokenizer.eos_token}"
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len, padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0).clone(),
        }


def load_dataset(name, split, limit=None):
    if name.endswith(".jsonl"):
        from datasets import load_dataset
        ds = load_dataset("json", data_files=name, split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Frankenstein-Labs/Cortex-MoE-565M")
    parser.add_argument("--dataset", default="legmlai/openhermes-fr")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="cortex_moe_sft")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--tiny-tokenizer", default="tiny_tokenizer")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    if args.tiny:
        from cortex_moe.configuration_cortex_moe import CortexMoeConfig
        from cortex_moe.modeling_cortex_moe import CortexMoeForCausalLM
        cfg = CortexMoeConfig(
            vocab_size=50257, n_positions=128, n_ctx=128,
            n_embd=64, n_layer=2, n_head=4,
            num_experts=4, num_experts_per_tok=2,
        )
        model = CortexMoeForCausalLM(cfg)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    if args.lora:
        if not PEFT_OK:
            raise RuntimeError("pip install peft")
        target_modules = []
        for n, _ in model.named_parameters():
            if "c_attn" in n or "c_proj" in n or "c_fc" in n or "mlp" in n:
                target_modules.append(n.rsplit(".", 1)[0])
        target_modules = sorted(set(target_modules))
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    ds = load_dataset(args.dataset, args.split, args.limit)
    train = InstructionDataset(ds, tokenizer, args.max_len)

    targs = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"SFT model saved to {args.output}")


if __name__ == "__main__":
    main()