# Chat local Cortex

Cette démo ajoute une interface de discussion Gradio au dépôt. Elle charge le modèle depuis les fichiers locaux; elle ne télécharge pas les poids à l’exécution et n’envoie pas les messages à un service distant.

## Prérequis

- Python 3.10 ou plus récent;
- Git et Git LFS;
- environ 2 Go de mémoire disponible pour la variante GPT‑2 de 124 M paramètres (davantage est préférable).

## Installation depuis GitHub

```bash
git lfs install
git clone https://github.com/Frankenstein-Labs/Cortex.git
cd Cortex
git lfs pull
python -m venv .venv
```

Activez ensuite l’environnement virtuel.

**Linux/macOS :**

```bash
source .venv/bin/activate
```

**Windows PowerShell :**

```powershell
.\.venv\Scripts\Activate.ps1
```

Installez PyTorch pour processeur uniquement, puis les dépendances de l’application :

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Lancez le chat :

```bash
python app.py
```

Ouvrez ensuite [http://127.0.0.1:7860](http://127.0.0.1:7860). Pour fermer le serveur, utilisez `Ctrl+C` dans le terminal.

## Réglages

- `CORTEX_MODEL_DIR` permet d’indiquer un autre dossier contenant `config.json`, le tokenizer et les poids;
- `GRADIO_SERVER_NAME` modifie l’adresse d’écoute (par défaut `127.0.0.1`, accès local seulement);
- `PORT` modifie le port (par défaut `7860`).

Ne remplacez l’adresse locale par `0.0.0.0` que si vous comprenez que cela peut rendre l’application accessible depuis d’autres appareils du réseau. Cette démo ne fournit pas d’authentification.

## Limites importantes

Le checkpoint est le GPT‑2 standard de 124 M de paramètres, entraîné à prédire la suite d’un texte, et non un assistant instruction-tuné. La mise en forme « User/Cortex » crée une interface de chat, mais ne transforme pas le modèle en assistant fiable. Attendez-vous à des réponses courtes, répétitives ou incorrectes; il est principalement associé à du texte anglais et peut être faible en français. N’utilisez pas ses réponses comme source fiable pour des décisions importantes.

Ce code est uniquement dans GitHub : GitHub n’exécute pas l’application dans la page du dépôt. Pour discuter avec Cortex, clonez le dépôt, récupérez les fichiers LFS et lancez `python app.py` sur votre ordinateur.
