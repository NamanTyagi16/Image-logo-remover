# Deploying LogoErase to Render

Your project is now fully configured for Render. All deployment blockers (large model rejection, missing requirements, port binding, and memory management) have been resolved.

---

## Quick Deploy Steps

### Step 1: Initialize Git and Push to GitHub
Open a terminal in this project folder and run:

```bash
git init
git add .
git commit -m "Configure project for Render deployment"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
git push -u origin main
```

*(Note: The 208MB `.onnx` model and virtual environment are automatically excluded from GitHub by `.gitignore`, so your push will succeed without hitting GitHub's 100MB limit).*

---

### Step 2: Deploy on Render

1. Go to [dashboard.render.com](https://dashboard.render.com/) and click **New +** -> **Web Service**.
2. Connect your GitHub repository.
3. Render will either auto-detect `render.yaml` or you can enter these settings manually:

| Setting | Value |
| :--- | :--- |
| **Language / Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt && python scripts/download_model.py` |
| **Start Command** | `uvicorn backend.app:app --host 0.0.0.0 --port $PORT` |
| **Health Check Path** | `/api/health` |

---

### Step 3: Environment Variables (Recommended for Free Tier)
Under **Environment Variables**, the following are configured:
* `PYTHON_VERSION`: `3.12.4`
* `MAX_PARALLEL_WORKERS`: `1` *(Keeps RAM usage safe on Render's 512MB free tier)*

---

## Local Development
Your local workflows remain untouched:
* Running `python backend/app.py` or double-clicking `Open LogoErase Preview.vbs` still starts locally on `127.0.0.1:8000`.
