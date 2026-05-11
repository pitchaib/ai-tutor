# AI Personal Tutor

**Gemma 4** is the core **thinking module**: it reads your textbook like a subject teacher, reasons about each concept on every page, and drives every helpful interactionùwhether the student types in **chat**, follows the **voice lesson**, asks a spoken doubt, or checks understanding with **auto-built quizzes**. The rest of the system (speech recognition, text-to-speech, PDF layout, and HTTP/Gradio UIs) wraps that intelligence so learning feels natural and bilingual.

Alongside the book, the tutor supports students from **Class 6 through Class 12** across **all Indian boards** (CBSE, ICSE, state boards) in **any subject** with a PDFùand in the studentùs preferred **voice and text** languages.

---

## Key Capabilities

- **Any class, any board, any subject** ù point it at any textbook PDF (Physics, Chemistry, Maths, Biology, Social Science, and more) for Class 6ù12.
- **Multilingual voice and text** ù speak and read explanations in English, Tamil, Hindi, Telugu, Kannada, Malayalam, and more; mix languages in one session.
- **Page-by-page teaching from the book** ù Gemma 4 turns raw PDF text into clear explanations; the student sees the page and hears follow-up audio where configured.
- **Interactive voice lesson** ù Gemma 4 scripts what the tutor says, asks check questions, interprets the studentùs spoken answers, and decides how to continue.
- **Contextual textbook chat** ù highlight text on the page and chat; Gemma 4 answers from the passage when it can, and still helps with on-topic follow-ups.
- **Auto-generated MCQs** ù Gemma 4 designs practice questions from the teacher explanations, with difficulty that ramps up.
- **Curated resources** ù ranked links (videos, articles) with student voting.
- **Signup and personalisation** ù board, class, medium, and school captured at first launch.

---

## Gemma 4: how the thinking tutor helps the student

Gemma 4 is not a side featureùit is the **reasoning engine** behind the product.

1. **From the book to clear explanations**  
   The pipeline extracts text from the textbook PDF. Gemma 4 rewrites it as structured **teacher explanations**: simpler language, examples where useful, and continuity from one page section to the next. That is the baseline ùconcept from the bookù the student sees and hears.

2. **Chat that stays grounded in the lesson**  
   In **textbook chat**, the student can select a sentence or paragraph and ask anything. Gemma 4 uses that selection (and the broader page context) to answer **on the page** when the information is there, branch **beyond the page** for fair on-topic help, or gently redirect **off-topic** questions. So chat feels like talking to a tutor who actually opened the same book.

3. **Interaction, not a monologue**  
   In **voice lesson mode**, Gemma 4 drives the **dialogue**: greetings, narration, follow-up questions, evaluation of what the student said, short clarifications when they interrupt with a doubt, and feedback that moves the lesson forward. Speech-to-text turns their voice into text; **Gemini TTS** (and optional Tamil stacks) turn Gemmaùs replies back into speechùso the ùthinkingù stays Gemma-shaped even when the medium is voice.

4. **Practice that matches what was taught**  
   MCQs are **generated from the same explanations** Gemma 4 wrote for the page, so quizzes reinforce the concepts the student just studiedùnot generic trivia.

In production you typically deploy **Gemma 4 (or a compatible Gemma/Gemini model)** on a **Vertex AI online prediction endpoint** and point this app at that endpoint via `configs/vertex.env`. For **local or notebook** experiments, the repo also references Hugging Face Gemma 4 model IDs in `teacher_pdf_pipeline.py`; the running services use the Vertex client path by default.

---

## Architecture

Three services run together, launched by a single script (`start.sh`):

| Service | Port | Description |
|---------|------|-------------|
| **learn_api** | 8000 | FastAPI backend ù page explanations, MCQ generation, textbook chat |
| **gradio_ui** | 7860 | Gradio Tutor UI ù voice lesson mode, resource voting |
| **html_frontend** | 8080 | HTML / Jinja2 frontend ù signup, textbook view, voice Q&A |

```
Browser
  |
  +-- html_frontend (port 8080)
        |
        +-- /api/*  -->  learn_api (port 8000)
        |                  |
        |                  +-- teacher_pdf_pipeline  -->  Vertex AI (Gemma 4 / Gemma / Gemini)
        |                  +-- assessment_pipeline   -->  Vertex AI (Gemma 4 / Gemma / Gemini)
        |                  +-- Tamil TTS             -->  IndicF5 (local GPU, optional)
        |                  +-- English TTS           -->  Gemini TTS (Vertex AI)
        |
        +-- /app    -->  gradio_ui (port 7860)
                           |
                           +-- voice_qa_pipeline  -->  Google Cloud Speech (ASR)
                                                   -->  Vertex AI (Gemma 4 / Gemma / Gemini)
                                                   -->  Gemini TTS (Vertex AI)
```

---

## Models and cloud services

This stack centres on a **Vertex-deployed text model** (recommended: **Gemma 4** on an online prediction endpoint for the behaviour described above). It also uses **Gemini TTS** for most spoken output and **Google Cloud Speech-to-Text** for student audio. The tables below map each piece to the modules.

### Gemma 4 / Vertex LLM (core reasoning)

Deploy Gemma 4 (or another compatible model) on Vertex; the app calls it only through `VertexEndpointClient` (`predict` / dedicated REST). Optional **Gemma 4** Hugging Face checkpoints (`google/gemma-4-26B-A4B`, `google/gemma-4-E4B`) in `teacher_pdf_pipeline.py` support **local / notebook** workflowsùthe live FastAPI and Gradio paths expect the **Vertex endpoint** you configure in `vertex.env`.

| Module | Role of the LLM (Gemma 4) |
|--------|---------------------------|
| **teacher_module** | Turn PDF sections into teacher-style explanations, structure chapters, and fill the teaching session. |
| **assessment_module** | Produce strict-JSON MCQs (four options, answer, explanation) from cached teacher text. |
| **ui_module** (`learn_api.py`) | Power `/textbook/chat` (with source tags), refresh page explanations, orchestrate quizzes. |
| **voice_qa_module** | Answer spoken doubts, evaluate student replies, greeting lines, and every other `generate_text` step in the voice loop. |

### Text-to-speech (TTS)

| Model / stack | Where it runs | Role by module |
|-----------------|---------------|----------------|
| **Gemini TTS** ù `gemini-2.5-flash-preview-tts` (Vertex, `google-genai`, prebuilt voice e.g. `Kore`) | Region from `VERTEX_TTS_LOCATION` / project from `VERTEX_TTS_PROJECT` | **voice_qa_module**: `synthesize_answer()` turns tutor script into WAV. **ui_module** Gradio (`voice_qa_page.py`): same for lesson audio. **ui_module_html** (`server.py`): tutor routes; static phrases cached on disk. |
| **IndicF5** (Tamil, `configs/config.yaml` + `requirements.txt`) | Local GPU recommended | Tamil reference-voice / batch setups ù complementary when you extend the pipeline. |

Configure TTS region separately from the main LLM endpoint so moving Gemma 4 to another region does not break voice output.

### Speech-to-text (STT / ASR)

| API and mode | Notes |
|--------------|--------|
| **Google Cloud Speech-to-Text** | `SpeechClient` uses **batch** `recognize()` on the **full** WAV after recording ù not **StreamingRecognize**. |
| **Recognition model** | Default `latest_short` ù tuned for short student utterances (roughly under ~60 seconds). |

| Module | Role of STT |
|--------|-------------|
| **voice_qa_module** | `transcribe_audio()` for answers, doubts, greetings. |
| **ui_module** (`voice_qa_page.py`) | Browser-captured audio into `run_lesson_step`. |
| **ui_module_html** (`server.py`) | Normalize uploads, then same transcription path. |

**ùStreamingù in the UI:** the **browser** may capture audio with silence detection; **server-side** recognition is one-shot per clip, not streaming STT.

---

## Folder Structure

```
Tutor/
|-- start.sh                              # Start / stop / restart / status all services
|-- requirements.txt                      # Consolidated Python dependencies
|-- configs/
|   |-- vertex.env.example               # Endpoint config template (copy to vertex.env)
|   +-- config.yaml                      # Tamil TTS (IndicF5) settings
+-- modules/
    |-- teacher_module/
    |   +-- src/
    |       +-- teacher_pdf_pipeline.py  # PDF parsing, Vertex client, session caching
    |
    |-- assessment_module/
    |   +-- src/
    |       +-- assessment_mcq_pipeline.py  # MCQ generation and caching
    |
    |-- voice_qa_module/
    |   +-- src/
    |       +-- voice_qa_pipeline.py     # ASR -> retrieval -> LLM -> TTS pipeline
    |   +-- scripts/
    |       +-- run_voice_qa.py          # CLI runner for voice pipeline
    |
    |-- ui_module/
    |   +-- src/
    |       |-- learn_api.py             # FastAPI backend app (port 8000)
    |       |-- learn_tab_app.py         # Gradio UI app (port 7860)
    |       |-- signup_page.py           # Student signup form
    |       +-- voice_qa_page.py         # Gradio interactive lesson page
    |
    +-- ui_module_html/
        |-- src/
        |   +-- server.py               # FastAPI HTML server (port 8080)
        +-- templates/                  # Jinja2 HTML templates
            |-- _layout.html
            |-- signup.html
            |-- welcome.html
            |-- textbook.html
            |-- practice.html
            |-- resources.html
            +-- voice.html
```

---

## Supported Boards and Classes

| Scope | Details |
|-------|---------|
| **Classes** | 6th through 12th standard |
| **Boards** | CBSE, ICSE / ISC, Tamil Nadu, Maharashtra, Karnataka, Kerala, Andhra Pradesh, Telangana, West Bengal, Rajasthan, UP, Gujarat, Bihar, and any other board |
| **Subjects** | Physics, Chemistry, Mathematics, Biology, Social Science, and any subject with a PDF textbook |
| **Languages (UI & text)** | English, Tamil, Hindi, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Odia, Punjabi, Assamese, Urdu, Sanskrit |
| **Languages (voice ASR)** | English (en-US), Tamil (ta-IN); additional Google Cloud Speech locales can be added |
| **Languages (TTS)** | English (Gemini TTS), Tamil (IndicF5); mix modes available |

---

## Prerequisites

- Python 3.10+
- A GCP project with:
  - **Vertex AI** API enabled
  - A deployed **online prediction endpoint** (recommended: **Gemma 4** or compatible Gemma/Gemini)
  - **Google Cloud Speech-to-Text API** enabled
  - **Application Default Credentials (ADC)** set up:
    ```bash
    gcloud auth application-default login
    ```
- NVIDIA GPU recommended for local IndicF5 Tamil TTS (CPU fallback is slow)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/pitchaib/ai-tutor.git
cd ai-tutor
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Install GPU-compatible PyTorch first (adjust cu124 to match your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install everything else
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
VERTEX_LOCATION="us-central1"          # region where your endpoint is deployed
VERTEX_ENDPOINT_ID="your-endpoint-id"  # from the Vertex AI console
VERTEX_API_ENDPOINT="..."              # dedicated PSC host, or leave blank for public
VERTEX_TTS_LOCATION="us-central1"     # region for Gemini TTS
VERTEX_TTS_PROJECT="your-gcp-project-id"
```

> `configs/vertex.env` is `.gitignore`d ù never commit it.

### 5. Point to your textbook PDF

```bash
export PDF_PATH="/path/to/your/textbook.pdf"
```

Any standard school textbook in PDF format works. The tutor reads it page by page.

### 6. Start all services

```bash
./start.sh
```

Open **http://127.0.0.1:8080** in your browser, complete the student signup, and start learning.

---

## Service Commands

```bash
./start.sh            # start all three services
./start.sh stop       # stop all services
./start.sh restart    # restart all services
./start.sh status     # show which services are running
./start.sh doctor     # diagnose Vertex AI endpoint connectivity
```

---

## Environment Variables

All credentials and paths are injected via environment variables sourced from `configs/vertex.env` by `start.sh`. You can also export them manually before running.

| Variable | Required | Description |
|----------|----------|-------------|
| `VERTEX_PROJECT_ID` | Yes | GCP project ID |
| `VERTEX_PROJECT_NUMBER` | PSC only | GCP project number |
| `VERTEX_LOCATION` | Yes | Endpoint region (e.g. `us-central1`) |
| `VERTEX_ENDPOINT_ID` | Yes | Online prediction endpoint ID |
| `VERTEX_ENDPOINT_URL` | Optional | Full console URL (alternative to the four vars above) |
| `VERTEX_API_ENDPOINT` | PSC only | Dedicated prediction hostname |
| `VERTEX_TTS_LOCATION` | Yes | Region for Gemini TTS |
| `VERTEX_TTS_PROJECT` | Yes | GCP project for Gemini TTS |
| `AITUTOR_ROOT` | Optional | Absolute path to this repo (auto-detected if not set) |
| `PDF_PATH` | Optional | Path to textbook PDF |
| `LEARN_API_URL` | Optional | URL of learn_api (default: `http://127.0.0.1:8000`) |
| `GRADIO_URL` | Optional | URL of Gradio UI (default: `http://127.0.0.1:7860`) |
| `HTML_PORT` | Optional | HTML server port (default: `8080`) |
| `GRADIO_PORT` | Optional | Gradio UI port (default: `7860`) |
| `API_PORT` | Optional | learn_api port (default: `8000`) |

---

## Key API Endpoints (learn_api, port 8000)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/learn/init?chapter_name=...` | Load the first page of a chapter |
| `POST` | `/learn/page` | Get AI explanation for a specific page |
| `POST` | `/learn/navigate` | Navigate to the previous or next page |
| `POST` | `/textbook/chat` | Contextual Q&A on selected text from the page |
| `GET` | `/learn/page_image?page_no=1` | Render a PDF page as a PNG image |
| `GET` | `/quiz?chapter_name=...&current_page=...` | Interactive MCQ quiz for a page |
| `POST` | `/quiz/submit` | Submit quiz answers and get results |

---

## Notes

- **Pre-populate the teacher cache** ù run the pipeline against your PDF once before first use. This stores AI-generated explanations in `modules/teacher_module/outputs/` so pages load faster without calling the LLM every time.
- **Tamil TTS (IndicF5)** downloads model weights (~2 GB) on first run to `model_cache/`. An internet connection is required on first startup.
- **English TTS** uses `gemini-2.5-flash-preview-tts` via Vertex AI. Ensure this model is available in your `VERTEX_TTS_LOCATION`.
- **Adding more languages** ù the ASR language code (e.g. `hi-IN` for Hindi) can be passed per session. TTS for additional languages can be integrated by extending `voice_qa_pipeline.py`.
