# Cortex-MoE

**Cortex-MoE** est une architecture de modèle de langage
**Mixture-of-Experts (MoE) multimodal nouvelle génération**,
conçue par **Abdoulaye Coumbassa / Frankenstein-Labs**.

Le modèle est construit par **sparse upcycling** à partir de
GPT-2 : les poids du FFN dense de chaque couche sont copiés dans
plusieurs experts, et un routeur apprend à sélectionner les
meilleurs experts pour chaque token. Une tour visuelle légère
et un projecteur permettent le traitement d'images.

## Architecture

| Composant | Détail |
| --- | --- |
| Backbone | Transformer post-norm style GPT-2 (12 couches, 768 dim, 12 têtes) |
| MoE | 8 experts par couche, top-2, routage softmax normalisé |
| Routeur | `Linear(768, 8)` + perte d'équilibrage Switch + z-loss |
| Tour visuelle | ViT léger (6 blocs pré-norm), patches 16×16, 224×224 |
| Projecteur | MLP 2 couches, sous-échantillonnage à 64 tokens d'image |
| Paramètres | **565M total / 181M actifs par token** (ratio 0,32) |

## Installation

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

## Conversion GPT-2 → Cortex-MoE

```bash
python3 convert_gpt2_to_moe.py --source gpt2 --out cortex_moe_565m
```

Le script préserve bit à bit les poids d'attention, des
layer-norms et des embeddings, duplique le FFN dense dans les
experts, et initialise les routeurs à zéro-naissance.

## Entraînement

```bash
python3 train_cortex_moe.py --model cortex_moe_565m --tiny
```

## Chat local

```bash
CORTEX_MODEL_DIR=cortex_moe_565m python3 app.py
```

## Dépôt Hugging Face

https://huggingface.co/Frankenstein-Labs/Cortex-MoE-565M

## Licence

- Code et architecture : [Cortex License v1.0](LICENSE)
- Poids dérivés de GPT-2 (OpenAI, licence MIT modifiée) —
  voir [NOTICE](NOTICE).
