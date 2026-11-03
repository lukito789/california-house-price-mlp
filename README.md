# Estimasi Harga Rumah California — MLP Regresi

Struktur: `frontend/` (HTML/CSS/JS) · `backend/` (FastAPI + Keras) · DB SQLite (`backend/app.db`, otomatis dibuat).

## Cara run
1. Download `housing.csv` dari Kaggle (camnugent/california-housing-prices) → taruh di `data/housing.csv`
2. `cd backend`
3. `python -m venv venv` lalu aktifkan (`venv\Scripts\activate` di Windows / `source venv/bin/activate` di Mac/Linux)
4. `pip install -r requirements.txt`
5. `python train.py`   (melatih model → `backend/artifacts/`)
6. `uvicorn app:app --reload`
7. Buka http://127.0.0.1:8000
