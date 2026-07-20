# Deploying Hybrid Vision AI

This is a **Flask app that runs real ML inference** (YOLOv8 + EfficientNet‑B0
via PyTorch). That shapes which platforms are actually a good fit — a couple
of quick, honest notes before you deploy anywhere:

## Platform fit

| Platform | Works free? | Why |
|---|---|---|
| **Hugging Face Spaces (Docker SDK)** | ✅ Recommended | Free CPU Basic tier gives **2 vCPU / 16GB RAM** — plenty of headroom for torch + torchvision + ultralytics + two loaded models. Purpose-built for exactly this kind of app. |
| **Render** | ⚠️ Free tier too small | Free web service is only **512MB RAM / 0.1 vCPU** — not enough for this stack; it OOMs/crash-loops on boot. Render's paid Standard tier (2GB RAM, $25/mo) works well, but there's no free tier that fits. |
| **Railway / Fly.io / Google Cloud Run** | ⚠️ Free allowance, not free forever | Free credits/quotas exist but are usage-metered and can require a card on file; fine as a fallback if HF Spaces doesn't suit you. |
| **A VPS (e.g. Hetzner, DigitalOcean Droplet) + Docker** | 💰 Paid | Full control, no cold starts, no RAM ceiling — but not free. |
| **Vercel / Netlify** | ❌ Not a fit | Serverless-functions only — no persistent process, small deployment size limits, short execution timeouts. PyTorch + Ultralytics blow past the size limit before your code even runs. |

**In short:** this app needs a persistent server process with enough RAM to
hold two loaded models in memory. Hugging Face Spaces' free CPU tier is the
best zero-cost fit for that today.

## Option A — Hugging Face Spaces (recommended, free)

1. Create a free account at huggingface.co (no card required).
2. Go to **huggingface.co/new-space** → give it a name → SDK: **Docker** →
   Visibility: your choice → Create Space.
3. Clone the empty Space repo it creates, copy this project's files into it
   (this repo already includes the `README.md` with the required
   `sdk: docker` / `app_port: 7860` frontmatter and a `Dockerfile` that
   listens on port 7860 by default), then:
   ```bash
   git add -A
   git commit -m "Deploy hybrid vision app"
   git push
   ```
4. The Space builds and starts automatically — watch the **Logs** tab.
   Once running, your app is live at `https://<your-username>-<space-name>.hf.space`.
5. Models load lazily on the first `/image` or `/video` request (not at
   boot), so the Space comes up fast; the first real request just takes
   ~10-20s longer while it loads once.

Note: free Spaces can go idle and take a few seconds to wake up after a
period of no traffic — same tradeoff as any free tier, but with far more
RAM to actually run this app reliably once awake.

## Option B — Render (works, but needs a paid plan)

1. Push this repo to GitHub.
2. In the Render dashboard: **New → Blueprint**, point it at your repo.
   Render will read `render.yaml` and `Dockerfile` automatically.
3. **Change the instance type to Standard (2GB RAM)** in Settings →
   Instance Type — the free tier (512MB) will crash-loop on this app.
4. Once live, `/health` returns `{"status": "ok"}` — Render uses this as
   the health check.

## Option C — Any other Docker host (Railway, Fly.io, Cloud Run, a VPS)

```bash
docker build -t hybrid-vision .
docker run -p 7860:7860 -e PORT=7860 hybrid-vision
```

Push the image to your platform of choice, or point the platform's Git
integration at this repo — they'll all build from the same `Dockerfile`.
Make sure whatever instance/tier you pick has **at least ~2GB RAM**.

## Option D — Classic buildpack host (Heroku-style)

`Procfile` and `runtime.txt` are included for platforms that build from
`requirements.txt` directly instead of Docker:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 2
```

## Configuration

Copy `.env.example` to `.env` for local runs, or set the same variables in
your platform's dashboard:

- `CORS_ORIGINS` — comma-separated allowed origins, or `*`
- `MAX_CONTENT_LENGTH_MB` — upload size cap (lower this on small free tiers)
- `FLASK_DEBUG` — leave `false` in production

## Reducing memory footprint further

`requirements.txt` already installs CPU-only PyTorch wheels and leaves out
DeepFace/TensorFlow (an optional fallback path — see the comment at the
bottom of `requirements.txt` and `requirements-full.txt`). `utils.py` also
loads both models lazily (on first request, not at import) and caps native
thread pools to 1, which reduces peak RAM and avoids boot-time timeouts on
constrained hosts. If a host still runs out of memory, that's the next
thing to trim.

## Local run

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

