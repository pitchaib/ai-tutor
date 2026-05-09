# AI Personal Tutor

A voice-first, bilingual (English / Tamil) AI tutor that teaches Class-12 Physics page-by-page from a textbook PDF using a Vertex AI–hosted language model (Gemma / Gemini) for explanations, assessment, and interactive Q&A.

---

## Architecture

Three services run together behind a single script (`start.sh`):

| Service | Port | Description |
|---------|------|-------------|
| **learn_api** | 8000 | FastAPI backend — page-wise explanations, MCQ generation, textbook chat |
| **gradio_ui** | 7860 | Gradio Tutor UI — voice lesson mode, resource voting |
| **html_frontend** | 8080 | HTML / Jinja2 frontend — signup, textbook view, voice Q&A |

```
Browser  ???  html_frontend (8080)
                 ??? proxies /api/* ???  learn_api (8000)
                 ??? redirects /app  ???  gradio_ui (7860)

learn_api  ???  teacher_pdf_pipeline  ???  Vertex AI endpoint (Gemma)
                                       ???  IndicF5 TTS (Tamil)
                                       ???  Gemini TTS (English)
gradio_ui  ???  voice_qa_pipeline     ???  Google Cloud Speech (ASR)
                                       ???  Vertex AI endpoint (Gemma)
                                       ???  Gemini TTS
```

---

## Folder Structure

```
Tutor/
??? start.sh                          # Start / stop / restart all services
??? requirements.txt                  # Consolidated Python dependencies
??? configs/
?   ??? vertex.env.example            # Endpoint config template — copy to vertex.env
?   ??? config.yaml                   # Tamil TTS (IndicF5) settings
??? modules/
    ??? teacher_module/src/
    ?   ??? teacher_pdf_pipeline.py   # PDF parsing, Vertex client, session caching
    ??? assessment_module/src/
    ?   ??? assessment_mcq_pipeline.py  # MCQ generation and caching
    ??? voice_qa_module/src/
    ?   ??? voice_qa_pipeline.py      # ASR ? retrieval ? LLM ? TTS pipeline
    ??? ui_module/src/
    ?   ??? learn_api.py              # FastAPI app (port 8000)
    ?   ??? learn_tab_app.py          # Gradio app (port 7860)
    ?   ??? signup_page.py            # Gradio signup form
    ?   ??? voice_qa_page.py          # Gradio interactive lesson page
    ??? ui_module_html/
        ??? src/server.py             # FastAPI HTML server (port 8080)
        ??? templates/                # Jinja2 HTML templates
```

---

## Prerequisites

- Python 3.10+
- A GCP project with:
  - **Vertex AI** enabled
  - A deployed **online prediction endpoint** serving a Gemma or Gemini model
  - **Google Cloud Speech-to-Text API** enabled
  - **Application Default Credentials (ADC)** configured:
    ```bash
    gcloud auth application-default login
    ```
- NVIDIA GPU recommended (for local IndicF5 Tamil TTS)

---

## Setup

### 1. Clone and enter the repo

```bash
cd Tutor
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Install GPU-compatible PyTorch first (adjust cu124 to your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
pip install -r requirements.txt
```

### 4. Configure the Vertex AI endpoint

```bash
cp configs/vertex.env.example configs/vertex.env
```

Edit `configs/vertex.env` and fill in your values:

```bash
VERTEX_PROJECT_ID="your-gcp-project-id"
VERTEX_PROJECT_NUMBER="123456789012"
VERTEX_LOCATION="us-central1"          # region of your deployed endpoint
VERTEX_ENDPOINT_ID="your-endpoint-id"  # from the Vertex AI console
VERTEX_API_ENDPOINT="..."              # dedicated PSC host, or leave blank for public
VERTEX_TTS_LOCATION="us-central1"
VERTEX_TTS_PROJECT="your-gcp-project-id"
```

> `configs/vertex.env` is `.gitignore`d — never commit it.

### 5. Provide a textbook PDF

Place your PDF at the path expected by `learn_api.py` or override it:

```bash
export PDF_PATH="/path/to/your/textbook.pdf"
```

### 6. Start all services

```bash
./start.sh
```

Open `http://127.0.0.1:8080` in your browser.

---

## Service Commands

```bash
./start.sh            # start all services
./start.sh stop       # stop all services
./start.sh restart    # restart all services
./start.sh status     # show running / stopped status
./start.sh doctor     # diagnose Vertex AI endpoint connectivity
```

---

## Environment Variables

All sensitive config is injected via environment variables. Set them in `configs/vertex.env` (sourced by `start.sh`) or export them manually before running.

| Variable | Required | Description |
|----------|----------|-------------|
| `VERTEX_PROJECT_ID` | Yes | GCP project ID |
| `VERTEX_PROJECT_NUMBER` | PSC only | GCP project number |
| `VERTEX_LOCATION` | Yes | Endpoint region (e.g. `us-central1`) |
| `VERTEX_ENDPOINT_ID` | Yes | Online prediction endpoint ID |
| `VERTEX_ENDPOINT_URL` | Optional | Full console URL (alternative to split vars) |
| `VERTEX_API_ENDPOINT` | PSC only | Dedicated prediction hostname |
| `VERTEX_TTS_LOCATION` | Yes | Region for Gemini TTS |
| `VERTEX_TTS_PROJECT` | Yes | Project for Gemini TTS |
| `AITUTOR_ROOT` | Optional | Absolute path to this repo (default: auto-detected) |
| `PDF_PATH` | Optional | Path to textbook PDF |
| `LEARN_API_URL` | Optional | URL of learn_api if not on localhost (default: `http://127.0.0.1:8000`) |
| `GRADIO_URL` | Optional | URL of Gradio UI (default: `http://127.0.0.1:7860`) |
| `HTML_PORT` | Optional | HTML server port (default: `8080`) |
| `GRADIO_PORT` | Optional | Gradio port (default: `7860`) |
| `API_PORT` | Optional | learn_api port (default: `8000`) |

---

## Key API Endpoints (learn_api, port 8000)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/learn/init?chapter_name=Electrostatics` | Load first page of chapter |
| `POST` | `/learn/page` | Get explanation for a specific page |
| `POST` | `/learn/navigate` | Navigate previous / next page |
| `POST` | `/textbook/chat` | Contextual Q&A on selected text |
| `GET` | `/learn/page_image?page_no=1` | Render page as PNG |
| `GET` | `/quiz?chapter_name=...&current_page=...` | Interactive MCQ quiz page |
| `POST` | `/quiz/submit` | Submit quiz answers |

---

## Notes

- The **teacher explanation cache** lives in `modules/teacher_module/outputs/` — pre-populate it by running the pipeline against your PDF before first use for faster page loads.
- **Tamil TTS** (IndicF5) downloads model weights on first run to `model_cache/`; this requires internet access and ~2 GB of storage.
- **English TTS** uses `gemini-2.5-flash-preview-tts` via the Vertex AI Generative AI API — ensure it is available in your `VERTEX_TTS_LOCATION`.
