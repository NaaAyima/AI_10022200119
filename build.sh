#!/usr/bin/env bash
set -e

echo "Installing ultra-lightweight CPU PyTorch to prevent memory crashing on Free Tier..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing remaining requirements..."
pip install -r requirements.txt

echo "Pre-downloading AI Embedding weights so Streamlit doesn't freeze on start..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "Build complete. App is ready for instant launch!"
