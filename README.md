# CaneScan - Sugarcane Leaf Disease Detection Website

FastAPI backend + SQLAlchemy/SQLite + plain HTML/CSS/JS frontend, serving your EfficientNetB0 model.

## What changed in this version (stabilisation update)

| Problem | Root cause | Fix |
|---|---|---|
| App could not start when TensorFlow failed to import | `predictor.py` imported TensorFlow at module load and `main.py` loaded the model at import/startup with no error handling | TensorFlow is imported only inside one function (`_build_predictor`). Any failure is recorded as a status; the API, database and website still start |
| No way to tell "API alive" from "model usable" | single health route | `/api/health` (alive, db ok, model_ready) and `/api/ready` (200 ready / 503 with a reason) |
| Prediction on a broken model | - | `/api/predict` returns **503** with a clear message. It never returns a fake or default label |
| Model path depended on a fixed file name | hard-coded name | Path is found from the backend folder; the final model is preferred, a single other `.keras/.h5` is accepted, ambiguity is reported |
| Wrong model/classes silently accepted | no checks | Input shape (RGB, 4-D) and output size vs `class_names.json` are verified, plus a warm-up prediction |
| Frontend folder not found on Linux/WSL | `frontend` vs `Frontend` case | Both spellings are detected |
| Mixed error formats | FastAPI defaults | Every error is `{"detail": "<message>", "code": "<machine code>"}`; no stack traces are sent to users |
| Frontend assumed the server was always fine | - | Readiness banner on the Scan page, friendly offline/503 messages, `config.js` for the API URL |
| One big requirements file | - | `requirements-api.txt`, `requirements-ml.txt`, `requirements-dev.txt` |

## The Windows "Application Control policy has blocked this file" error

`ImportError: DLL load failed while importing _dtypes: An Application Control policy has blocked this file`
(Code Integrity events 3033 / 3089) is **Windows security refusing to load `tensorflow\python\framework\_dtypes.pyd`**.
It is not a missing package, so reinstalling TensorFlow will not help, and no application code can or should work around it.

What the code does about it: detects it, reports `tensorflow_blocked_by_policy` on `/api/ready`, keeps everything
except prediction working, and shows a banner on the website.

What actually fixes it (pick one):
1. **Ask the device administrator** to allow the TensorFlow files (or the Python environment) in the Application Control /
   WDAC / Smart App Control policy. Give them the output of:
   ```powershell
   Get-WinEvent -LogName "Microsoft-Windows-CodeIntegrity/Operational" -MaxEvents 50 |
     Where-Object { $_.Id -in 3033,3077,3089 } | Format-List TimeCreated, Id, Message
   ```
2. **Run the backend on Linux** (WSL2 Ubuntu, if your administrator allows WSL, or a Linux server/Docker/cloud host). Windows
   Code Integrity does not govern Linux binaries inside WSL2.
3. Use a different machine or hosting service for the backend; the website only needs the API URL (`frontend/config.js`).

A new virtual environment on the same Windows machine will most likely be blocked the same way, because the policy
applies to the files, not to the environment.

## Setup - Windows PowerShell (existing Python 3.11 conda env as the base)

```powershell
cd D:\Sugarcane-Web\Backend

# private virtual environment (nothing is installed globally)
C:\Users\Admin\anaconda3\envs\rasterio_env\python.exe -m venv .venv

# 1) API only - the website, history, guide and health checks work with just this
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt

# 2) check whether TensorFlow can run on this machine (read-only, changes nothing)
.\.venv\Scripts\python.exe -m pip install tensorflow==2.16.2
.\.venv\Scripts\python.exe diagnose_tf.py

# 3) start the server
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000 (website) and http://127.0.0.1:8000/docs (API docs).
If step 2 says `tensorflow_blocked_by_policy`, use the options above. Use the TensorFlow version you trained with
(`pip show tensorflow` in the training environment) and edit `requirements-ml.txt` to match.

## Setup - Ubuntu / WSL2 (viable alternative)

```bash
# copy the project onto the Linux filesystem (the original on D: is left untouched)
cp -r /mnt/d/Sugarcane-Web ~/Sugarcane-Web
cd ~/Sugarcane-Web/Backend          # or backend, whichever your folder is called

sudo apt update && sudo apt install -y python3-venv python3-pip     # Ubuntu 24.04 ships Python 3.12
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt     # API + TensorFlow
python diagnose_tf.py
uvicorn main:app --host 0.0.0.0 --port 8000
```
Then open http://localhost:8000 in your Windows browser.

## Your model files
Put these in `backend/model/` (they are only read, never modified):
- the trained `.keras` model (the final `sugarcane_disease_efficientnetB0.keras` is preferred automatically)
- `class_names.json` from the same training run

Your existing `app.db` and `uploads/` are kept: startup only creates missing tables and never drops data.
`disease_info.json` is the source for the `diseases` table and is re-applied at every start.

## Run the tests
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -v
```
Tests use a temporary database and folders, so they never touch your real data.
- `tests/test_predictor_unit.py` - error classification (including the exact Windows message), class names, model discovery, failure reporting. The last test runs **real inference** with your real model and is skipped automatically if the model or TensorFlow is unavailable.
- `tests/test_api.py` - health/readiness, 503 without a model, upload validation, database persistence. Tests marked "stub" use a stand-in predictor only to test validation and storage; they do not prove the model works.

## Settings (environment variables, all optional)
| Variable | Default | Meaning |
|---|---|---|
| DATABASE_URL | SQLite `backend/app.db` | e.g. `postgresql+psycopg2://user:pass@host/db` |
| MODEL_DIR / MODEL_PATH / CLASS_NAMES_PATH | `backend/model/...` | custom locations |
| LOAD_MODEL_ON_STARTUP | true | `false` = load the model on the first prediction instead |
| CONFIDENCE_THRESHOLD | 0.70 | below this the site says "not sure" |
| MAX_UPLOAD_MB | 5 | upload size limit |
| ALLOWED_ORIGINS | * | comma-separated; set to your domain in production |

## API
- GET `/api/health` - alive? `{status, database_ok, model_ready}`
- GET `/api/ready` - 200 if predictions work, else 503 `{ready:false, code, message, hint}`
- POST `/api/predict` (multipart field `file`) - 200 result, 400/413 bad image, 503 model unavailable, 500 inference failed
- GET `/api/history`, POST `/api/predictions/{id}/feedback`, GET `/api/diseases`, GET `/api/stats`

## Docker
```bash
docker build -t canescan .
docker run -p 8000:8000 -v canescan_data:/app/backend/uploads canescan
```
