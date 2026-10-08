# Cortex

**Cortex** est le projet de modèles multimodaux
**Mixture-of-Experts nouvelle génération** d'Abdoulaye Coumbassa
/ **Frankenstein-Labs**.

Le modèle phare, **Cortex-MoE-565M**, est un MoE multimodal
construit par sparse upcycling à partir de GPT-2 :
**565M paramètres totaux, 181M actifs par token** (8 experts,
top-2), avec une tour visuelle ViT légère.

## Démarrage rapide

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
python3 convert_gpt2_to_moe.py --source gpt2 --out cortex_moe_565m
CORTEX_MODEL_DIR=cortex_moe_565m python3 app.py
```

Voir [CORTEX_MOE.md](CORTEX_MOE.md) pour l'architecture et
[CORTEX_CHAT.md](CORTEX_CHAT.md) pour l'application de chat.

## Modèle sur Hugging Face

https://huggingface.co/Frankenstein-Labs/Cortex-MoE-565M

## Licence

- Code et architecture : [Cortex License v1.0](LICENSE)
- Poids dérivés de GPT-2 (OpenAI, licence MIT modifiée) —
  voir [NOTICE](NOTICE).
