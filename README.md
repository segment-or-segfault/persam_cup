# PerSAM-CUP: Personalized Segment Anything for the Cups Dataset

This project adapts [Personalize-SAM](https://github.com/segment-anything/segment-anything) to the CUPS segmentation dataset. It includes:
- a full modified segmentation pipeline
- DINOv2 + SAM feature matching
- cascade refinement
- our **Adaptive Mask Fusion** module
- a lightweight React/Vite UI
- evaluation scripts + pre-generated outputs

Both backend and frontend run together through a single `make run`.

---

## Project Structure

```
persam_cup/
├── Makefile
├── README.md
├── requirements.txt
├── run.sh
├── backend/
│   └── Personalize-SAM-main/
│       ├── data/
│       ├── eval/
│       ├── outputs/
│       │   └── results_cups.csv
│       └── sam_vit_h_4b8939.pth  # SAM ViT-H checkpoint
└── frontend/
    └── cupSAM-webui-main/        # React + Vite UI
└── venv/                        # created automatically
```

---

## Backend Pipeline

The full pipeline is contained in:
```bash
backend/Personalize-SAM-main/app.py
```

It implements:
1. Input reading and pre-processing
2. DINOv2 feature extraction
3. SAM feature extraction
4. Similarity maps (DINO + SAM)
5. Point selection
6. Cascade refinement
7. Adaptive Mask Fusion
8. Fallback logic
9. Final mask upsampling and return

The `/segment` endpoint returns a PNG mask for each query image.

---

## Setup & Running

### 1. Make sure `run.sh` is executable (only once):
```bash
chmod +x run.sh
```

### 2. Start everything (backend + frontend):
```bash
make run
```

`make run` will:
- install frontend dependencies
- download the SAM ViT-H checkpoint if missing
- create a Python virtual environment
- install backend dependencies
- start the backend server
- start the frontend Vite dev server

Frontend will open at:
```
http://localhost:8080/
```

---

## Cleaning Python Caches

To remove all `__pycache__` folders:
```bash
make clean
```

---

## Evaluation

All predicted masks for each method are stored in:
```bash
backend/outputs/
```

To evaluate everything (compute IoU, count IoU ≥ 0.7, write CSV):
```bash
cd backend/eval
python3 evaluate_iou.py
```

This script automatically scans all folders under `outputs/cups/` and saves:
```bash
results_cups.csv
```

---

## Notes
- Use **Node 20** (Node 22 causes Vite NAPI errors)
- The SAM checkpoint is large, so it is downloaded only once
- Only `make run` and `make clean` are needed for normal use
