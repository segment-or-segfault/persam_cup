#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend/cupSAM-webui-main"

MODEL_NAME="sam_vit_h_4b8939.pth"
MODEL_URL="https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth"
MODEL_PATH="$BACKEND_DIR/$MODEL_NAME"

# Frontend install
cd "$FRONTEND_DIR"
npm install

# Backend model download
if [ ! -f "$MODEL_PATH" ]; then
    curl -L "$MODEL_URL" -o "$MODEL_PATH"
fi

# Backend venv
cd "$PROJECT_ROOT"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
pip3 install --upgrade pip
pip3 install -r "requirements.txt"

# Run backend
./venv/bin/python backend/Personalize-SAM-main/app.py &
BACKEND_PID=$!

# Run frontend
cd "$FRONTEND_DIR"
npm run dev

# Cleanup backend on exit
kill $BACKEND_PID 2>/dev/null || true
