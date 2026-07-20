---
title: Hybrid Vision AI
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Hybrid Vision AI

YOLOv8 + EfficientNet-B0 hybrid object detection/classification pipeline,
served through a Flask web app (image and video upload, class filtering,
and multi-object tracking for video).

Deployed here on Hugging Face Spaces' free CPU tier (2 vCPU / 16GB RAM),
which comfortably fits this app's memory footprint (torch + torchvision +
ultralytics + two loaded models). See `DEPLOY.md` for deployment notes and
other platform options.

Models load lazily on the first `/image` or `/video` request rather than at
boot, so the app starts and responds to `/health` immediately — the first
real request just takes a little longer (~10-20s) while it loads once.
