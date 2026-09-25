---
title: MOSAIC EDA
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
license: mit
short_description: Agents that clean, explore, and fact-check any dataset
---

# MOSAIC EDA

**Multimodal Orchestrated System for Analysis, Inspection & Cleaning**

A crew of AI agents, built with [CrewAI](https://www.crewai.com/), that cleans, explores, and
fact-checks any dataset: tables, text, images, audio, and video, or a zip of them.

> Status: tables and image datasets work end to end, with review and revision loops
> (Phase 4). Audio, text, and video are next.
> The full design is in [`docs/MOSAIC-EDA-Blueprint.pdf`](docs/MOSAIC-EDA-Blueprint.pdf).

## How it works (short version)

- A **CrewAI Flow** routes each job. Plain code does every calculation, and agents plan,
  interpret, and explain.
- Every claim an agent makes cites an **evidence artifact**, and code checks the numbers.
- Cleaning uses only an **allowed list of operations**, and the app exports a
  `cleaning_pipeline.py` you can rerun.
- Runs free on the Gemini free tier with two **model pools** (Flash-Lite for most work, Flash
  for review) and automatic fallback.

## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env          # then add your GEMINI_API_KEY
python -m mosaic.llm.key_check   # one test request per model
python app.py                 # http://127.0.0.1:7860
```

Or with Docker:

```bash
docker build -t mosaic-eda .
docker run -p 7860:7860 --env-file .env mosaic-eda
```

## Tests

```bash
pytest              # unit tests, no API calls
pytest -m live      # also calls the real Gemini API (uses free-tier quota)
ruff check . && ruff format --check .
```

## Credits

Illustrations: [Highlights](https://www.highlights.design/) by Outdraw Design (CC0).
