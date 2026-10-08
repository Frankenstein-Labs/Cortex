"""Fine-tune the vision tower + projector of Cortex-MoE on image/text pairs.

The text backbone is frozen. Only the ViT blocks, the post-LN and the
2-layer projector are trained, so the MoE weights stay untouched.

Dataset: any HF dataset with `image` + `text` columns. COCO captions
(`mscoco/coco_captions`) or a French one (`wikimedia/wikipedia` with
images) work. Pass `--image-column text` for caption-only datasets.

Usage:
  python train_vision.py --dataset mscoco/coco_captions \\
      --image-column image --text-column captions \\
      --output cortex_moe_vision
"""

import argparse
import json
import os

import torch
import cortex_moe  # noqa: registers "cortex_moe" in Auto* classes
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


class ImageTextDataset(Dataset):
    def __init__(self, data, tokenizer, image_size=224, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        text = item["text"]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len, return_tensors="pt",
        )
        return {
            "pixel_values": image,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": enc["input_ids"].squeeze(0).clone(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Frankenstein-Labs/Cortex-MoE-565M")
    parser.add_argument("--dataset", default="mscoco/coco_captions")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output", default="cortex_moe_vision")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
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

    # freeze everything except the vision tower and projector
    for name, p in model.named_parameters():
        p.requires_grad = (
            name.startswith("vision_tower")
            or name.startswith("vision_projector")
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.2f}M / {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    train = ImageTextDataset(ds, tokenizer, max_len=args.max_len)

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
    print(f"Vision-tuned model saved to {args.output}")


if __name__ == "__main__":
    main()