"""Démo locale de chat pour le checkpoint Cortex/GPT-2."""
from __future__ import annotations

import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path(os.environ.get("CORTEX_MODEL_DIR", Path(__file__).resolve().parent)).expanduser()
MODEL_LABEL = "Cortex · GPT-2 base"
_MODEL_LOCK = threading.Lock()


def _is_lfs_pointer(path: Path) -> bool:
    """Return True when a model file is still an unhydrated Git LFS pointer."""
    try:
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, Any, int]:
    """Load the local tokenizer and model once; never download weights at runtime."""
    config_path = MODEL_DIR / "config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"Configuration introuvable dans {MODEL_DIR}. "
            "Lancez l’application depuis le dépôt Cortex ou définissez CORTEX_MODEL_DIR."
        )

    weight_candidates = [MODEL_DIR / "model.safetensors", MODEL_DIR / "pytorch_model.bin"]
    if not any(path.is_file() and not _is_lfs_pointer(path) for path in weight_candidates):
        raise RuntimeError(
            "Les poids ne sont pas présents localement. Depuis la racine du dépôt, lancez "
            "`git lfs install` puis `git lfs pull`, et relancez l’application."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model.to("cpu")
    model.eval()

    context_limit = int(
        getattr(model.config, "n_positions", None)
        or getattr(model.config, "max_position_embeddings", None)
        or 1024
    )
    return tokenizer, model, context_limit


def _message_text(content: Any) -> str:
    """Normalize Gradio message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(part for part in parts if part)
    return ""


def build_prompt(history: list[dict[str, Any]], message: str) -> str:
    """Format a short chat transcript as plain text for the base GPT-2 model."""
    lines = [
        (
            "The following is a text conversation with Cortex, a GPT-2 language model. "
            "Continue the conversation naturally.\n"
        )
    ]
    # Keep the latest turns; token truncation below provides a second context-window guard.
    for item in history[-12:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = _message_text(item.get("content", "")).strip()
        if not text or role not in {"user", "assistant"}:
            continue
        speaker = "User" if role == "user" else "Cortex"
        lines.append(f"{speaker}: {text}\n")
    lines.append(f"User: {message.strip()}\nCortex:")
    return "".join(lines)


def reply(
    message: str,
    history: list[dict[str, Any]],
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Generate a continuation locally; this is not an instruction-tuned assistant."""
    message = (message or "").strip()
    if not message:
        return "Écrivez un message pour commencer."

    tokenizer, model, context_limit = load_model()
    max_new_tokens = max(16, min(int(max_new_tokens), 160, context_limit - 1))
    max_input_tokens = max(1, context_limit - max_new_tokens)
    prompt = build_prompt(history or [], message)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    input_length = int(inputs["input_ids"].shape[-1])
    if input_length == 0:
        return "Je n’ai pas pu préparer ce message. Essayez de le reformuler."

    with _MODEL_LOCK, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(0.2, min(float(temperature), 1.5)),
            top_p=0.92,
            repetition_penalty=1.12,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0, input_length:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Stop if the base model starts inventing the next speaker turn.
    answer = re.split(r"\n(?:User|Human|Cortex)\s*:", answer, maxsplit=1)[0].strip()
    answer = re.sub(r"^(?:Assistant|Cortex)\s*:\s*", "", answer, flags=re.IGNORECASE)
    return answer or "Je n’ai pas généré de réponse cette fois. Essayez un autre réglage ou une autre question."


demo = gr.ChatInterface(
    fn=reply,
    type="messages",
    title=MODEL_LABEL,
    description=(
        "Démo locale utilisant les poids présents dans ce dépôt. "
        "GPT‑2 est un modèle de continuation de texte de 124 M de paramètres, "
        "non fine-tuné pour suivre des consignes : ses réponses peuvent être faibles, "
        "répétitives ou inexactes, surtout en français. Les messages ne sont pas envoyés "
        "à un service distant par cette application."
    ),
    chatbot=gr.Chatbot(type="messages", height=520, label="Conversation"),
    textbox=gr.Textbox(placeholder="Écrivez à Cortex…", autofocus=True),
    additional_inputs=[
        gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Température"),
        gr.Slider(32, 160, value=96, step=16, label="Longueur maximale de réponse (tokens)"),
    ],
    theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
)


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        show_error=True,
    )
