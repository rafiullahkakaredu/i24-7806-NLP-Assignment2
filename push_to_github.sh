#!/usr/bin/env bash
# Creates the git history (8 incremental commits) and pushes to GitHub.
# Usage (from inside this folder, after copying the Colab outputs in):
#     bash push_to_github.sh <your-github-username>
# Needs: git. Optional: GitHub CLI `gh` (logged in with `gh auth login`) to create the repo automatically.
set -e
USER_NAME="${1:-rafiullahkakaredu}"
REPO="i24-7806-NLP-Assignment2"
NB=$(ls *.ipynb 2>/dev/null | head -1)

[ -d .git ] && { echo ".git already exists – remove it first if you want a fresh history"; exit 1; }
git init -q && git branch -M main
mkdir -p embeddings models data figures scripts
for d in embeddings models data figures; do [ -z "$(ls -A $d 2>/dev/null)" ] && touch $d/.gitkeep; done

c() { git add "$@" 2>/dev/null || true; git diff --cached --quiet || git commit -q -m "$MSG"; echo "✓ $MSG"; }
MSG="Project skeleton: README, requirements, .gitignore, folder layout"; c README.md requirements.txt .gitignore push_to_github.sh embeddings/.gitkeep models/.gitkeep data/.gitkeep figures/.gitkeep
MSG="Setup: Urdu normaliser, tokeniser, A1 stemmer/lemmatiser, corpus + metadata loader"; c scripts/a2_part0_setup.py scripts/run_all.py
MSG="Part 1: TF-IDF, PPMI, t-SNE, Skip-gram Word2Vec with negative sampling, analogies, 4-condition MRR comparison"; c scripts/a2_part1_embeddings.py
MSG="Part 2: rule-based POS tagger + gazetteer NER, BiLSTM-CRF with Viterbi, conlleval, ablations A1-A4"; c scripts/a2_part2_sequence_labeling.py
MSG="Part 3: Transformer encoder from scratch (attention, MHSA, FFN, sinusoidal PE, Pre-LN, CLS) + BiLSTM comparison"; c scripts/a2_part3_transformer.py
MSG="Report generator and executed Colab notebook"; c scripts/a2_part4_report.py ${NB:+"$NB"}
MSG="Add generated artefacts: embeddings, annotated CoNLL data, trained models, figures"; c embeddings data models figures
MSG="Add final report (report.pdf) and results.json"; c report.pdf results.json
git log --oneline

URL="https://github.com/$USER_NAME/$REPO"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh repo create "$REPO" --public --source . --remote origin --push
else
  echo; echo "Create an EMPTY public repo named $REPO at https://github.com/new (no README), then press Enter."
  read -r _
  git remote add origin "$URL.git"
  git push -u origin main
fi
echo "Done → $URL"
