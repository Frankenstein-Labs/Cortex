# Cortex Chat

Application de chat locale pour **Cortex-MoE** (texte et image).

## Lancer

```bash
CORTEX_MODEL_DIR=cortex_moe_565m python3 app.py
```

## Utilisation

- Saisissez un message texte pour continuer la conversation.
- Collez le **chemin d'un fichier image** pour le décrire
  (prétraitement ImageNet intégré, aucune dépendance
  torchvision requise).
- Tapez `quit` ou `exit` pour quitter.

## Options

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | `cortex_moe_565m` | Dossier du modèle (ou variable `CORTEX_MODEL_DIR`) |
| `--max-new-tokens` | `128` | Longueur maximale de la réponse |
| `--seed` | `42` | Graine aléatoire |

## Dépôt Hugging Face

https://huggingface.co/Frankenstein-Labs/Cortex-MoE-565M
