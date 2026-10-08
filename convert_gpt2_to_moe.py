"""Convert GPT-2 weights to Cortex-MoE via sparse upcycling.

- Attention, layernorms and embeddings are preserved bit-for-bit.
- Each dense FFN is copied into all experts (sparse upcycling).
- Routers are freshly initialized.
"""

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cortex_moe.configuration_cortex_moe import CortexMoeConfig
from cortex_moe.modeling_cortex_moe import CortexMoeForCausalLM


def convert(source: str, out_dir: str, num_experts: int, top_k: int):
    os.makedirs(out_dir, exist_ok=True)
    print(f"loading {source} ...")
    gpt2 = AutoModelForCausalLM.from_pretrained(source).eval()
    cfg = gpt2.config

    config = CortexMoeConfig(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.n_positions,
        n_ctx=cfg.n_ctx,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_inner=cfg.n_inner,
        activation_function=cfg.activation_function,
        resid_pdrop=cfg.resid_pdrop,
        embd_pdrop=cfg.embd_pdrop,
        layer_norm_epsilon=cfg.layer_norm_epsilon,
        initializer_range=cfg.initializer_range,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
    )
    print(
        f"building Cortex-MoE: {num_experts} experts, top-{top_k}, "
        f"{cfg.n_layer} layers"
    )
    model = CortexMoeForCausalLM(config)

    state = gpt2.state_dict()
    # transformers >= 5 exposes keys with the "transformer." prefix
    state = {
        k[len("transformer.") :] if k.startswith("transformer.") else k: v
        for k, v in state.items()
    }
    new_state = {}
    for key, value in state.items():
        if key == "lm_head.weight":
            continue
        new_key = "transformer." + key
        if new_key.endswith(".attn.bias"):
            continue
        new_state[new_key] = value

    for i in range(config.n_layer):
        mlp_w1 = state[f"h.{i}.mlp.c_fc.weight"]
        mlp_b1 = state[f"h.{i}.mlp.c_fc.bias"]
        mlp_w2 = state[f"h.{i}.mlp.c_proj.weight"]
        mlp_b2 = state[f"h.{i}.mlp.c_proj.bias"]
        for e in range(num_experts):
            new_state[f"transformer.h.{i}.moe.experts.{e}.c_fc.weight"] = mlp_w1.clone()
            new_state[f"transformer.h.{i}.moe.experts.{e}.c_fc.bias"] = mlp_b1.clone()
            new_state[f"transformer.h.{i}.moe.experts.{e}.c_proj.weight"] = mlp_w2.clone()
            new_state[f"transformer.h.{i}.moe.experts.{e}.c_proj.bias"] = mlp_b2.clone()

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("transformer.h.")]
    expected_missing = {
        "lm_head.weight",
        "vision_tower.patch_embed.weight",
        "vision_tower.patch_embed.bias",
        "vision_tower.cls_token",
        "vision_tower.pos_embed",
        "vision_tower.proj.weight",
        "vision_tower.proj.bias",
        "vision_projector.proj.0.weight",
        "vision_projector.proj.0.bias",
        "vision_projector.proj.2.weight",
        "vision_projector.proj.2.bias",
    }
    for i in range(config.n_layer):
        expected_missing.add(f"transformer.h.{i}.moe.router.gate.weight")
    for b in range(config.vision_depth):
        expected_missing.update(
            {
                f"vision_tower.blocks.{b}.ln_1.weight",
                f"vision_tower.blocks.{b}.ln_1.bias",
                f"vision_tower.blocks.{b}.attn.in_proj_weight",
                f"vision_tower.blocks.{b}.attn.in_proj_bias",
                f"vision_tower.blocks.{b}.attn.out_proj.weight",
                f"vision_tower.blocks.{b}.attn.out_proj.bias",
                f"vision_tower.blocks.{b}.ln_2.weight",
                f"vision_tower.blocks.{b}.ln_2.bias",
                f"vision_tower.blocks.{b}.mlp.0.weight",
                f"vision_tower.blocks.{b}.mlp.0.bias",
                f"vision_tower.blocks.{b}.mlp.2.weight",
                f"vision_tower.blocks.{b}.mlp.2.bias",
            }
        )
    expected_missing.update(
        {
            "vision_tower.ln_post.weight",
            "vision_tower.ln_post.bias",
        }
    )
    real_missing = [k for k in missing if k not in expected_missing]
    if real_missing:
        raise RuntimeError(f"missing keys: {real_missing}")
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected}")
    model.lm_head.weight = model.transformer.wte.weight

    total = sum(p.numel() for p in model.parameters())
    active = total
    for i in range(config.n_layer):
        for _, p in model.transformer.h[i].moe.experts.named_parameters():
            active -= p.numel() * (num_experts - top_k) / num_experts
    print(f"total params: {total/1e6:.1f}M | active per token: {active/1e6:.1f}M")

    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.save_pretrained(out_dir)

    shutil.copy("LICENSE", os.path.join(out_dir, "LICENSE"))
    shutil.copy("NOTICE", os.path.join(out_dir, "NOTICE"))
    if os.path.exists("model_card.md"):
        shutil.copy("model_card.md", os.path.join(out_dir, "README.md"))

    summary = {
        "model_type": "cortex_moe",
        "source": source,
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "params_total": total,
        "params_active_per_token": active,
    }
    with open(os.path.join(out_dir, "moe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="gpt2")
    parser.add_argument("--out", default="cortex_moe_565m")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    convert(args.source, args.out, args.num_experts, args.top_k)
