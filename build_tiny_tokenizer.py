"""Build a minimal tokenizer (vocab_size=1000) for the tiny Cortex-MoE demo.

Uses a byte-level BPE tokenizer with a single dummy merge so that any
text encodes to at least one token. The actual tokens are byte
sequences (GPT-2 byte-to-unicode mapping), so the tokenizer is
functional even though the model is randomly initialized.
"""

import json
import os

from transformers import GPT2TokenizerFast

PAD = "<|pad|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
UNK = "<|unk|>"


def build(path, vocab_size=1000):
    os.makedirs(path, exist_ok=True)
    # GPT-2 byte-level vocab: 256 bytes + 3 specials = 259 base tokens
    base_vocab = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
    for i in range(256):
        base_vocab[f"byte{i}"] = 4 + i
    # pad to vocab_size with dummy tokens
    for i in len(base_vocab), vocab_size:
        base_vocab[f"tok{i}"] = i
    merges = [["byte0", "byte1"]]  # single dummy merge so BPE can run
    tok = GPT2TokenizerFast(
        vocab_file=None,
        merges_file=None,
        vocab=base_vocab,
        merges=merges,
        unk_token=UNK,
        bos_token=BOS,
        eos_token=EOS,
        pad_token=PAD,
        add_prefix_space=False,
    )
    tok.save_pretrained(path)
    print(f"saved tokenizer to {path} (vocab={tok.vocab_size})")
    enc = tok("Quelle est la capitale de la France ?", max_length=32, truncation=True)
    print("sample ids:", enc["input_ids"])


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tiny_tokenizer")
    p.add_argument("--vocab", type=int, default=1000)
    args = p.parse_args()
    build(args.out, args.vocab)