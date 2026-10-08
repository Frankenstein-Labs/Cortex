"""Placeholder training loop for Cortex-MoE.

Loads a converted checkpoint and runs a few optimizer steps on a
synthetic batch, to verify gradients flow through the router,
the experts and the vision tower.
"""

import argparse

import torch
from torch.utils.data import DataLoader, Dataset

from cortex_moe.configuration_cortex_moe import CortexMoeConfig
from cortex_moe.modeling_cortex_moe import CortexMoeForCausalLM


class ImagenetDummy(Dataset):
    def __init__(self, n=16, seq_len=128, image_size=224):
        self.n = n
        self.seq_len = seq_len
        self.image_size = image_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        input_ids = torch.randint(0, 50257, (self.seq_len,))
        labels = input_ids.clone()
        pixel_values = torch.randn(3, self.image_size, self.image_size)
        return {"input_ids": input_ids, "labels": labels, "pixel_values": pixel_values}


def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    return {"input_ids": input_ids, "labels": labels, "pixel_values": pixel_values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cortex_moe_565m")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--tiny", action="store_true", help="override config with tiny sizes")
    args = parser.parse_args()

    if args.tiny:
        config = CortexMoeConfig(
            vocab_size=1000,
            n_positions=128,
            n_ctx=128,
            n_embd=64,
            n_layer=2,
            n_head=4,
            num_experts=4,
            num_experts_per_tok=2,
        )
        model = CortexMoeForCausalLM(config)
    else:
        model = CortexMoeForCausalLM.from_pretrained(args.model)

    loader = DataLoader(
        ImagenetDummy(image_size=model.config.image_size),
        batch_size=args.batch_size,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    print("training OK")


if __name__ == "__main__":
    main()
