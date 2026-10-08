---
license: other
license_name: cortex-license-v1.0
library_name: transformers
tags:
  - mixture-of-experts
  - moe
  - gpt2
  - multimodal
  - cortex
  - frankenstein-labs
---

# Cortex-MoE-565M

**Cortex-MoE-565M** est un modèle de langage multimodal Mixture-of-Experts
construit par **Abdoulaye Coumbassa / Frankenstein-Labs**, dérivé de
GPT-2 base via **sparse upcycling**.

- Dépôt GitHub : https://github.com/Frankenstein-Labs/Cortex
- Licence : [Cortex License v1.0](https://github.com/Frankenstein-Labs/Cortex/blob/main/LICENSE)

## Architecture

| Composant | Détail |
| --- | --- |
| Backbone texte | Transformer post-norm style GPT-2 (12 couches, 768 dim, 12 têtes, ctx 1024) |
| MoE | 8 experts par couche, routage top-2 normalisé |
| Routeur | `Linear(768, 8)` ; perte d'équilibrage Switch + z-loss ; capacité 1,25 |
| Tour visuelle | ViT léger (6 blocs pré-norm, 768 dim), patches 16×16, 224×224 |
| Projecteur | MLP 2 couches + sous-échantillonnage à 64 tokens d'image |
| Paramètres | **565M total / 181M actifs par token** (ratio 0,32) |

## Origine des poids

- Attention, layer-norms, embeddings : **poids GPT-2 préservés bit à bit**
  (GPT-2 est distribué par OpenAI sous licence MIT modifiée).
- Experts : FFN dense de GPT-2 copié dans les 8 experts de chaque couche.
- Routeurs : initialisation aléatoire (neufs).
- Tour visuelle et projecteur : initialisation aléatoire (neufs).

## Utilisation

```python
import cortex_moe  # enregistre l'architecture "cortex_moe" dans Auto*
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Frankenstein-Labs/Cortex-MoE-565M")
model = AutoModelForCausalLM.from_pretrained("Frankenstein-Labs/Cortex-MoE-565M")

# Texte seul
inputs = tokenizer("The capital of France is", return_tensors="pt")
generated = model.generate(**inputs, max_new_tokens=32)

# Multimodal (pixel_values : (B, 3, 224, 224), normalisé ImageNet)
import torch
pixel_values = torch.randn(1, 3, 224, 224)
generated = model.generate(**inputs, pixel_values=pixel_values, max_new_tokens=32)
```

Ou avec l'application de chat locale :

```bash
CORTEX_MODEL_DIR=. python app.py   # depuis le dossier du modèle téléchargé
```

## Limites importantes

- **La tour visuelle est aléatoire** : ce checkpoint est un upcycling de
  GPT-2, pas un modèle vision-langue entraîné. Les réponses fondées sur
  les images ne sont pas significatives tant que le modèle n'est pas
  affiné sur des données image-texte (`train_cortex_moe.py` dans le
  dépôt GitHub).
- Les experts partagent le même FFN initial : la spécialisation n'apparaît
  qu'avec l'entraînement.
- C'est un modèle de **continuation de texte** (GPT-2 base), pas un
  assistant instruction-tuné : réponses potentiellement répétitives ou
  inexactes, surtout en français. Ne pas utiliser comme source fiable
  pour des décisions importantes.

## Entraînement

Le modèle est obtenu par conversion déterministe de
[gpt2](https://huggingface.co/gpt2) (sparse upcycling), sans phase
d'entraînement supplémentaire. Le script de conversion et le script
d'affinage sont dans le dépôt :
https://github.com/Frankenstein-Labs/Cortex

## Licence

Code et architecture : Cortex License v1.0 © 2026 Abdoulaye Coumbassa /
Frankenstein-Labs. Poids dérivés de GPT-2 (© OpenAI, licence MIT
modifiée) — la notice d'OpenAI et la licence MIT doivent être conservées
dans toute redistribution des poids dérivés.
