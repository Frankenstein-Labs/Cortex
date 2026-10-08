# Évaluation honnête du pipeline Cortex-MoE

## 1. SFT (Supervised Fine-Tuning) en français

Le script `sft_cortex_moe.py` a été testé sur un modèle **tiny** (2 couches, 64 dim, 4 experts, vocab 50257) avec le dataset `sft_dataset.jsonl` (20 paires instruction/réponse en français, 5 époques, batch 4, max-len 128).

- **Loss d'entraînement** : 10.59 (converge, le pipeline Trainer fonctionne)
- **Modèle tiny** : 2.1M paramètres, trop petit pour apprendre des réponses factuelles

### Résultats avant/après (tiny, 24 tokens max)

| Question | BASE (non entraîné) | SFT (5 époques) |
| --- | --- | --- |
| Quelle est la capitale de la France ? | `JapaneseJapanese...` (répétition) | `?\n\n\n...` (s'arrête) |
| Qui a écrit Les Misérables ? | `HeartThreadThread...` (répétition) | `\n\n\n...` (s'arrête) |
| Combien de continents... | `shady Fail Fail...` (répétition) | `? Intermediate...` (répétition) |

**Interprétation** : le SFT modifie effectivement le comportement (le modèle tiny arrête de répéter le même token et produit des structures différentes), mais 2,1M paramètres sur 20 exemples ne suffisent pas à retenir les réponses. Sur le modèle **565M** (GPU requis, ~12 GB VRAM), le SFT avec un dataset d'instructions français (ex. `legmlai/openhermes-fr`) doit donner des réponses correctes.

## 2. Tour visuelle

Le script `train_vision.py` gèle le backbone texte et entraîne uniquement `vision_tower` + `vision_projector` sur des paires image-texte. Aucune image n'est disponible dans le sandbox (4 CPU, 11 Go RAM, pas de GPU) — le script est prêt pour un dataset COCO/VQA sur GPU.

## 3. Évaluation avant/après

Le script `eval_before_after.py` compare le modèle de base et le modèle SFT sur un jeu de questions (`eval_questions.jsonl`, 20 questions). La baseline (565M, non entraîné) donne des répétitions :

```
Q: Quelle est la capitale de la France ?
BASE: The capital of France is the capital of the French Republic, and the capital of the French Republic is the
```

Le modèle continue du texte (comportement GPT-2), pas une réponse directe. Après SFT sur un dataset instruction/réponse, il doit répondre "Paris".

## Pour aller plus loin (GPU requis)

```bash
# 1. SFT sur le modèle 565M
python sft_cortex_moe.py --dataset legmlai/openhermes-fr --output cortex_moe_sft --lora --epochs 3

# 2. Tour visuelle
python train_vision.py --dataset mscoco/coco_captions --image-column image --text-column captions --output cortex_moe_vision

# 3. Évaluation
python eval_before_after.py --base Frankenstein-Labs/Cortex-MoE-565M --ft cortex_moe_sft --questions eval_questions.jsonl
```